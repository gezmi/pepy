"""
Confidence file discovery and loading.

Supports:
- AF2/ColabFold: JSON files alongside PDB structures
- AF3: JSON files (full_data + summary_confidences) alongside CIF structures
- ChAI: NPZ files alongside PDB/CIF structures

To add a new prediction method, add entries to AF2_REPLACEMENTS or
AF3_PATTERNS / CHAI_PATTERNS, and handle loading in _load_file().
"""

import os
import glob
import numpy as np
import pandas as pd
import ujson as json
from typing import Optional, Dict, Any, List


# ============================================================================
# Naming patterns — edit these to add support for new prediction methods
# ============================================================================

# Substrings to strip from PDB filenames before converting to JSON path
AF2_STRIP_PATTERNS = [
    '_superimpos',
    '_superimposed',
    '_super',
    '_truncated',
    '_align_inter_pymol',
]

# Replacements applied to PDB filename to derive the AF2 JSON path
AF2_REPLACEMENTS = [
    ('.pdb', '.json'),
    ('_unrelaxed_', '_scores_'),
    ('_relaxed_', '_scores_'),
]

# Flexible key names for extracting metrics from JSON data
METRIC_KEYS = {
    'pae': ['predicted_aligned_error', 'pae', 'PAE'],
    'iptm': ['iptm', 'i_ptm', 'ipTM', 'iPTM'],
    'ptm': ['ptm', 'PTM', 'pTM', 'pTm'],
    'plddt': ['plddt', 'confidence', 'bfactor'],
}


# ============================================================================
# Public API
# ============================================================================

def load_confidence_data(structure_path: str,
                         confidence_file: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Load confidence data for a structure file.

    Discovers and loads confidence metrics (PAE matrix, iPTM, pTM) from
    prediction output files located alongside the structure.

    Args:
        structure_path: Absolute path to the structure file (.pdb or .cif)
        confidence_file: Explicit path to confidence file. If provided,
            skips auto-discovery.

    Returns:
        Dict with keys: pae_matrix (DataFrame or None), iptm (float or None),
        ptm (float or None), confidence (float or None),
        has_confidence_data (bool). Returns None if no confidence data found.
    """
    if confidence_file:
        if not os.path.exists(confidence_file):
            return None
        raw = _load_file(confidence_file)
        return _parse_metrics(raw)

    # Auto-discover confidence file(s)
    file_info = _find_confidence_files(structure_path)
    if not file_info:
        return None

    raw = _load_file(file_info['confidence_file'])

    # For AF3, merge in summary_confidences (contains iPTM/pTM)
    if file_info.get('summary_file'):
        try:
            with open(file_info['summary_file'], 'r') as f:
                summary = json.load(f)
            raw.update(summary)
        except Exception:
            pass

    return _parse_metrics(raw)


# ============================================================================
# File discovery
# ============================================================================

def _find_confidence_files(structure_path: str) -> Optional[Dict[str, str]]:
    """
    Find confidence file(s) for a structure.

    Returns dict with 'confidence_file' and optionally 'summary_file',
    or None if not found.
    """
    is_cif = structure_path.endswith('.cif')
    is_chai_pdb = 'idx' in os.path.basename(structure_path)

    if is_cif or is_chai_pdb:
        return _find_af3_or_chai(structure_path)
    else:
        return _find_af2(structure_path)


def _find_af2(pdb_path: str) -> Optional[Dict[str, str]]:
    """Find AF2/ColabFold JSON confidence file for a PDB structure."""
    json_path = pdb_path

    for pattern in AF2_STRIP_PATTERNS:
        json_path = json_path.replace(pattern, '')

    for old, new in AF2_REPLACEMENTS:
        json_path = json_path.replace(old, new)

    if os.path.isfile(json_path):
        return {'confidence_file': json_path}

    return None


def _find_af3_or_chai(structure_path: str) -> Optional[Dict[str, str]]:
    """Find AF3 or ChAI confidence files for a CIF/ChAI structure."""
    directory = os.path.dirname(structure_path) or '.'
    rank = _extract_rank(structure_path)

    # --- Try AF3 JSON files first ---

    # Direct path construction from structure filename
    if structure_path.endswith('.cif'):
        base_full_data = structure_path.replace('_model', '_full_data').replace('.cif', '.json')
        base_summary = structure_path.replace('_model', '_summary_confidences').replace('.cif', '.json')

        if os.path.isfile(base_full_data):
            return {
                'confidence_file': base_full_data,
                'summary_file': base_summary if os.path.isfile(base_summary) else None
            }

        # Try _confidences variant
        base_confidences = base_full_data.replace('_full_data', '_confidences')
        if os.path.isfile(base_confidences):
            return {
                'confidence_file': base_confidences,
                'summary_file': base_summary if os.path.isfile(base_summary) else None
            }

    # Glob fallback with rank
    if rank:
        for pattern in [f'{directory}/*full_data*{rank}*.json',
                        f'{directory}/*confidences*{rank}*.json']:
            matches = glob.glob(pattern)
            if matches:
                # Also look for summary
                summary_matches = glob.glob(f'{directory}/*summary*{rank}*.json')
                return {
                    'confidence_file': matches[0],
                    'summary_file': summary_matches[0] if summary_matches else None
                }

    # --- Try ChAI NPZ files ---
    if rank:
        npz_patterns = [
            f'{directory}/*{rank}*.npz',
            f'{directory}/*{rank}*.npy',
        ]
    else:
        npz_patterns = [
            f'{directory}/*.npz',
        ]

    for pattern in npz_patterns:
        matches = glob.glob(pattern)
        if matches:
            return {'confidence_file': matches[0]}

    return None


def _extract_rank(file_path: str) -> Optional[str]:
    """Extract rank/model number from filename."""
    basename = os.path.basename(file_path)
    name = basename.split('.')[0]

    # Look for numeric parts in underscore-separated segments (from end)
    if '_' in name:
        for part in reversed(name.split('_')):
            if part.isdigit():
                return part

    # Fallback: last digit character
    for char in reversed(name):
        if char.isdigit():
            return char

    return None


# ============================================================================
# File loading
# ============================================================================

def _load_file(filepath: str) -> Dict[str, Any]:
    """Load confidence data from JSON or NPZ file."""
    if filepath.endswith('.npz') or filepath.endswith('.npy'):
        return _load_npz(filepath)
    elif filepath.endswith('.json'):
        return _load_json(filepath)
    else:
        raise ValueError(f"Unsupported confidence file format: {filepath}")


def _load_json(json_path: str) -> Dict[str, Any]:
    """Load JSON confidence file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def _load_npz(npz_path: str) -> Dict[str, Any]:
    """Load ChAI NPZ/NPY confidence file."""
    data = np.load(npz_path)

    result = {}

    if 'pae_scores' in data:
        result['pae_scores'] = data['pae_scores']
    else:
        # Fallback: average across models if 3D
        arrays = [data[key] for key in data.files if key != 'iptm']
        if arrays:
            arr = arrays[0]
            if arr.ndim > 2:
                arr = np.mean(arr, axis=0) if arr.ndim == 3 else arr[0]
            result['pae_scores'] = arr

    if 'iptm' in data:
        result['iptm'] = float(data['iptm'].flat[0])

    return result


# ============================================================================
# Metrics parsing
# ============================================================================

def _parse_metrics(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse raw confidence data into standardized metrics.

    Returns dict with: pae_matrix, iptm, ptm, confidence, has_confidence_data
    """
    # PAE matrix
    pae_matrix = None
    for key in METRIC_KEYS['pae']:
        if key in raw_data and raw_data[key] is not None:
            pae_array = np.array(raw_data[key])
            if pae_array.ndim == 3:
                pae_array = pae_array[0]
            if pae_array.ndim == 2:
                pae_matrix = pd.DataFrame(pae_array)
            break

    # Scalar metrics
    iptm = _extract_scalar(raw_data, METRIC_KEYS['iptm'])
    ptm = _extract_scalar(raw_data, METRIC_KEYS['ptm'])

    # Combined confidence
    confidence = None
    if iptm is not None and ptm is not None:
        confidence = round(0.8 * iptm + 0.2 * ptm, 2)

    return {
        'pae_matrix': pae_matrix,
        'iptm': iptm,
        'ptm': ptm,
        'confidence': confidence,
        'has_confidence_data': any([pae_matrix is not None, iptm is not None, ptm is not None])
    }


def _extract_scalar(data: Dict[str, Any], possible_keys: List[str]) -> Optional[float]:
    """Extract a scalar float from data using possible key names."""
    for key in possible_keys:
        if key in data and data[key] is not None:
            value = data[key]
            try:
                if hasattr(value, 'item'):
                    return float(value.item())
                elif isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0:
                    return float(value[0]) if len(value) == 1 else None
                else:
                    return float(value)
            except (ValueError, TypeError, IndexError):
                continue
    return None

"""
Confidence file discovery and loading utilities.
"""

import os
import glob
import numpy as np
import ujson as json
from typing import Optional, Dict, Any, Tuple
from pathlib import Path


class ConfidenceFileFinder:
	"""
	Systematic confidence file discovery for different prediction methods.

	Uses file format to determine prediction method:
	- PDB files → AF2/ColabFold
	- CIF files → AF3/ChAI
	"""

	def __init__(self):
		pass

	def find_confidence_files(self, structure_path: str, require_confidence: bool = True) -> Optional[Dict[str, Any]]:
		"""
		Find confidence files for a structure.

		Args:
			structure_path: Path to structure file (.pdb or .cif)
			require_confidence: Whether to raise error if files not found

		Returns:
			Dict with 'type', 'confidence_file', 'iptm_file' keys, or None

		Raises:
			FileNotFoundError: If require_confidence=True and files not found
			ValueError: If unsupported file format
		"""
		file_format = self._get_file_format(structure_path)

		if file_format == 'pdb' and 'idx' not in structure_path:
			return self._find_af2_colabfold_files(structure_path, require_confidence)
		elif file_format == 'cif' or 'idx' in structure_path:
			return self._find_af3_chai_files(structure_path, require_confidence)
		else:
			raise ValueError(f"Unsupported file format: {structure_path}")

	def _get_file_format(self, structure_path: str) -> str:
		"""Get file format from structure path."""
		if structure_path.endswith('.pdb'):
			return 'pdb'
		elif structure_path.endswith('.cif'):
			return 'cif'
		else:
			return 'unknown'

	def _find_af2_colabfold_files(self, pdb_path: str, require_confidence: bool) -> Optional[Dict[str, Any]]:
		"""Find AF2/ColabFold JSON files for PDB structures."""

		json_file = pdb_path
		json_file = json_file.replace('_superimpos', '')
		json_file = json_file.replace('_superimposed', '')
		json_file = json_file.replace('_super', '')
		json_file = json_file.replace('_truncated', '')
		json_file = json_file.replace('.pdb', '.json')
		json_file = json_file.replace('_unrelaxed_', '_scores_')
		json_file = json_file.replace('_relaxed_', '_scores_')

		if os.path.isfile(json_file):
			return {
				'type': 'AF2_ColabFold',
				'confidence_file': json_file,
				'iptm_file': None
			}
		elif require_confidence:
				raise FileNotFoundError(f"No AF2/ColabFold confidence file found for {pdb_path}")
		else:
			return None

	def _find_af3_chai_files(self, cif_path: str, require_confidence: bool) -> Optional[Dict[str, Any]]:
		"""Find AF3/ChAI files for CIF structures."""
		base = cif_path.replace('_model', '').replace('.cif', '')
		rank = self._extract_rank(cif_path)
		directory = os.path.dirname(cif_path) or '.'

		# Try ChAI first (NPZ files)
		chai_patterns = [
			f"{base}_{rank}.npz" if rank else f"{base}.npz",
			f"{directory}/*{rank}*.npz" if rank else f"{directory}/*.npz"
		]

		chai_file = self._try_patterns(chai_patterns)
		if chai_file:
			return {
				'type': 'ChAI',
				'confidence_file': chai_file,
				'iptm_file': None  # iPTM in same NPZ for ChAI
			}

		# Try AF3 (separate JSON files)
		af3_confidence_patterns = [
			f"{base}_full_data_{rank}.json" if rank else f"{base}_full_data.json",
			f"{base}_confidences_{rank}.json" if rank else f"{base}_confidences.json",
		]

		af3_iptm_patterns = [
			f"{base}_summary_confidences_{rank}.json" if rank else f"{base}_summary_confidences.json",
		]

		# Add glob patterns as fallback
		if rank:
			af3_confidence_patterns.extend([
				f"{directory}/*confidences*{rank}*.json",
				f"{directory}/*full_data*{rank}*.json"
			])
			af3_iptm_patterns.extend([
				f"{directory}/*summary*{rank}*.json"
			])

		confidence_file = self._try_patterns(af3_confidence_patterns)
		iptm_file = self._try_patterns(af3_iptm_patterns)

		if confidence_file:
			return {
				'type': 'AF3',
				'confidence_file': confidence_file,
				'iptm_file': iptm_file
			}

		if require_confidence:
			raise FileNotFoundError(f"No AF3/ChAI confidence file found for {cif_path}")
		return None

	def _extract_rank(self, file_path: str) -> Optional[str]:
		"""Extract rank/model number from file path."""
		basename = os.path.basename(file_path)

		# Common patterns for rank extraction
		if '_' in basename:
			parts = basename.split('_')
			for part in reversed(parts):
				# Remove file extension
				part = part.split('.')[0]
				if part.isdigit():
					return part

		# Try to find rank in the basename itself
		for char in reversed(basename.split('.')[0]):
			if char.isdigit():
				return char

		return None

	def _try_patterns(self, patterns: list) -> Optional[str]:
		"""Try a list of file patterns and return first existing file."""
		for pattern in patterns:
			# Handle glob patterns
			if '*' in pattern:
				matches = glob.glob(pattern)
				if matches:
					return matches[0]  # Return first match
			else:
				# Handle exact file paths
				if os.path.isfile(pattern):
					return pattern
		return None


class ConfidenceFileLoader:
	"""
	Load and parse confidence data from different file formats.
	"""

	@staticmethod
	def load_confidence_data(file_info: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Load confidence data based on file type.

		Args:
			file_info: Dict from ConfidenceFileFinder with type and file paths

		Returns:
			Dict with parsed confidence metrics
		"""
		if file_info['type'] == 'ChAI':
			return ConfidenceFileLoader._load_chai_npz(file_info['confidence_file'])
		elif file_info['type'] in ['AF2_ColabFold', 'AF3']:
			return ConfidenceFileLoader._load_af_json(file_info)
		else:
			raise ValueError(f"Unknown confidence file type: {file_info['type']}")

	@staticmethod
	def _load_chai_npz(npz_path: str) -> Dict[str, Any]:
		"""Load ChAI confidence data from NPZ file."""
		try:
			data = np.load(npz_path)

			# Extract PAE matrix
			if 'pae_scores' in data:
				pae_matrix = data['pae_scores']
			else:
				# Fallback: compute average if multiple entries
				arrays = [data[key] for key in data.files if key != 'iptm']
				if arrays:
					pae_matrix = np.mean(arrays, axis=2) if len(arrays[0].shape) > 2 else arrays[0]
				else:
					pae_matrix = None

			# Extract iPTM if available
			iptm = float(data.get('iptm', [0])[0]) if 'iptm' in data else None

			return {
				'pae_matrix': pae_matrix,
				'iptm': iptm,
				'ptm': None,  # ChAI typically doesn't have separate pTM
				'plddt': None  # Not in ChAI NPZ files
			}

		except Exception as e:
			raise RuntimeError(f"Failed to load ChAI NPZ file {npz_path}: {e}")

	@staticmethod
	def _load_af_json(file_info: Dict[str, Any]) -> Dict[str, Any]:
		"""Load AlphaFold JSON confidence data."""
		confidence_file = file_info['confidence_file']
		iptm_file = file_info.get('iptm_file')

		try:
			# Load main confidence file
			with open(confidence_file, 'r') as f:
				confidence_data = json.load(f)

			# Load iPTM file if separate (AF3)
			iptm_data = {}
			if iptm_file and os.path.isfile(iptm_file):
				with open(iptm_file, 'r') as f:
					iptm_data = json.load(f)

			# Combine data
			combined_data = {**confidence_data, **iptm_data}

			# Extract metrics with flexible key names
			result = {
				'pae_matrix': ConfidenceFileLoader._extract_metric(
					combined_data, ['predicted_aligned_error', 'pae', 'PAE']
				),
				'iptm': ConfidenceFileLoader._extract_metric(
					combined_data, ['iptm', 'i_ptm', 'ipTM', 'iPTM']
				),
				'ptm': ConfidenceFileLoader._extract_metric(
					combined_data, ['ptm', 'PTM', 'pTM', 'pTm']
				),
				'plddt': ConfidenceFileLoader._extract_metric(
					combined_data, ['plddt', 'confidence', 'bfactor']
				)
			}

			return result

		except Exception as e:
			raise RuntimeError(f"Failed to load AF JSON file {confidence_file}: {e}")

	@staticmethod
	def _extract_metric(data: Dict[str, Any], possible_keys: list) -> Any:
		"""Extract metric from data using possible key names."""
		for key in possible_keys:
			if key in data:
				return data[key]
		return None
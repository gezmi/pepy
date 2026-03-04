"""
Protein complex structure class for interface analysis.
"""

import warnings
import os
import numpy as np
from typing import Optional, List, Dict, Union, Tuple, Any

import pandas as pd
from biopandas.pdb import PandasPdb

from ..io.loaders import load_pdb_file, load_cif_file, fix_af3_file_safe, add_ca_indices
from ..core.base import CACHE_KEYS
from ..utils.validation import validate_file_path, validate_chains

warnings.filterwarnings("ignore")


class ProteinComplex(PandasPdb):
    """
    Extended biopandas structure class for protein complex interface analysis.

    Handles complexes with one or more binder chains and one or more receptor chains.
    Provides interface calculation, confidence data loading, and metrics computation.
    """

    def __init__(self):
        super().__init__()

        # Chain assignments (both sides are lists)
        self.binder_chains: List[str] = []
        self.receptor_chains: List[str] = []

        # Cache
        self._cache = {key: None for key in CACHE_KEYS.values()}
        self._cache['chain_lengths'] = {}

        # Interface results
        self.interface_residues_binder: Optional[List[int]] = None
        self.interface_residues_receptor: Optional[List[int]] = None
        self.interface_calculated: bool = False

        # Confidence data
        self.confidence_data: Optional[Dict[str, Any]] = None
        self.has_confidence_data: bool = False
        self._structure_file_path: Optional[str] = None

    @classmethod
    def from_file(cls, filepath: str, **kwargs) -> 'ProteinComplex':
        """
        Load structure from PDB or CIF file.

        Args:
            filepath: Path to structure file (.pdb or .cif)

        Returns:
            ProteinComplex instance
        """
        instance = cls()
        instance.load_structure(filepath, **kwargs)
        return instance

    def load_structure(self, filepath: str, fix_af3: bool = True) -> None:
        """
        Load structure from file and prepare for analysis.

        Args:
            filepath: Path to structure file
            fix_af3: Whether to fix AF3 CIF files (adds missing columns)
        """
        validate_file_path(filepath)
        self._structure_file_path = os.path.abspath(filepath)

        if filepath.endswith('.cif'):
            if fix_af3:
                fix_af3_file_safe(filepath)
            self._df = load_cif_file(filepath)
        elif filepath.endswith('.pdb'):
            self._df = load_pdb_file(filepath)

        if 'ATOM' in self._df and not self._df['ATOM'].empty:
            self._df['ATOM'] = add_ca_indices(self._df['ATOM'])

        self.clear_cache()

    def identify_chains(self,
                       binder_chains: Optional[Union[str, List[str]]] = None,
                       receptor_chains: Optional[List[str]] = None) -> None:
        """
        Identify binder and receptor chains in the complex.

        Logic:
        1. If binder_chains provided: use them, rest are receptors (or specified receptors)
        2. If only receptor_chains provided: remaining chain(s) become binder
        3. If nothing specified: shortest chain becomes binder

        Args:
            binder_chains: Binder chain ID(s) — single string or list
            receptor_chains: List of receptor chain IDs

        Raises:
            ValueError: If chain identification logic fails
        """
        available_chains = self.get_available_chains()

        # Normalize binder_chains to list
        if isinstance(binder_chains, str):
            binder_chains = [binder_chains]

        # Validate
        if binder_chains:
            for bc in binder_chains:
                if bc not in available_chains:
                    raise ValueError(f"Binder chain '{bc}' not found. Available: {available_chains}")
        if receptor_chains:
            for rc in receptor_chains:
                if rc not in available_chains:
                    raise ValueError(f"Receptor chain '{rc}' not found. Available: {available_chains}")
        if binder_chains and receptor_chains:
            overlap = set(binder_chains) & set(receptor_chains)
            if overlap:
                raise ValueError(f"Chain(s) {overlap} cannot be both binder and receptor")
        if not available_chains or len(available_chains) < 2:
            raise ValueError("Need at least two chains in structure")

        if binder_chains:
            self.binder_chains = binder_chains
            if receptor_chains:
                self.receptor_chains = receptor_chains
            else:
                self.receptor_chains = [c for c in available_chains if c not in binder_chains]

        elif receptor_chains:
            remaining = [c for c in available_chains if c not in receptor_chains]
            if len(remaining) == 0:
                raise ValueError("No chains left for binder after specifying receptors")
            elif len(remaining) == 1:
                self.binder_chains = remaining
                self.receptor_chains = receptor_chains
            else:
                raise ValueError(
                    f"Multiple non-receptor chains found: {remaining}. "
                    "Please specify binder_chains explicitly."
                )

        else:
            # Auto-detect: shortest chain is binder
            chain_lengths = self.get_chain_lengths()
            shortest_chain = min(chain_lengths.items(), key=lambda x: x[1])[0]
            self.binder_chains = [shortest_chain]
            self.receptor_chains = [c for c in available_chains if c != shortest_chain]

        self.clear_cache()
        print(f'Binder chain(s): {",".join(self.binder_chains)}, '
              f'receptor chain(s): {",".join(self.receptor_chains)}')

    def get_available_chains(self) -> List[str]:
        """Get list of available chain IDs in the structure."""
        if 'ATOM' not in self._df or self._df['ATOM'].empty:
            return []
        return sorted(self._df['ATOM']['chain_id'].unique().tolist())

    def get_chain_lengths(self) -> Dict[str, int]:
        """Get number of residues for each chain."""
        if not self._cache['chain_lengths']:
            if 'ATOM' not in self._df or self._df['ATOM'].empty:
                self._cache['chain_lengths'] = {}
            else:
                self._cache['chain_lengths'] = (
                    self._df['ATOM']
                    .groupby('chain_id')['residue_number']
                    .nunique()
                    .to_dict()
                )
        return self._cache['chain_lengths']

    def get_chain_info(self) -> Dict[str, Dict]:
        """Get comprehensive information about all chains."""
        info = {}
        chain_lengths = self.get_chain_lengths()

        for chain_id in self.get_available_chains():
            chain_atoms = self._df['ATOM'][self._df['ATOM']['chain_id'] == chain_id]
            info[chain_id] = {
                'length': chain_lengths[chain_id],
                'type': self._get_chain_type(chain_id),
                'residue_range': (
                    chain_atoms['residue_number'].min(),
                    chain_atoms['residue_number'].max()
                ),
                'sequence': self._get_chain_sequence(chain_id)
            }
        return info

    def _get_chain_type(self, chain_id: str) -> str:
        """Determine if chain is binder, receptor, or unassigned."""
        if chain_id in self.binder_chains:
            return 'binder'
        elif chain_id in self.receptor_chains:
            return 'receptor'
        return 'unassigned'

    def _get_chain_sequence(self, chain_id: str) -> str:
        """Get single-letter amino acid sequence for chain."""
        chain_atoms = self._df['ATOM'][self._df['ATOM']['chain_id'] == chain_id]
        if chain_atoms.empty:
            return ""

        ca_atoms = chain_atoms[chain_atoms['atom_name'] == 'CA'].sort_values('residue_number')
        if ca_atoms.empty:
            return ""

        temp_pdb = PandasPdb()
        temp_pdb._df = {'ATOM': ca_atoms, 'HETATM': self._df.get('HETATM', ca_atoms.iloc[:0].copy())}

        seq_df = temp_pdb.amino3to1()
        chain_seq = seq_df[seq_df['chain_id'] == chain_id]
        return ''.join(chain_seq['residue_name'].values)

    def is_ready_for_analysis(self) -> bool:
        """Check if structure is ready for analysis."""
        return (
            len(self.binder_chains) > 0 and
            len(self.receptor_chains) > 0 and
            'ATOM' in self._df and
            not self._df['ATOM'].empty
        )

    def get_analysis_summary(self) -> Dict:
        """Get summary of current structure and chain assignments."""
        if not self.is_ready_for_analysis():
            return {'status': 'not_ready', 'message': 'Chain identification required'}

        chain_info = self.get_chain_info()
        return {
            'status': 'ready',
            'binder_chains': self.binder_chains,
            'receptor_chains': self.receptor_chains,
            'binder_length': sum(chain_info[c]['length'] for c in self.binder_chains),
            'receptor_length': sum(chain_info[c]['length'] for c in self.receptor_chains),
            'total_chains': len(self.get_available_chains()),
            'unassigned_chains': [
                c for c in self.get_available_chains()
                if c not in self.binder_chains + self.receptor_chains
            ]
        }

    # ============================================================================
    # INTERFACE CALCULATION
    # ============================================================================

    def calculate_interface(self,
                           cb_cutoff: float = 8.0,
                           all_atom_cutoff: float = 4.0,
                           min_interface_size: int = 1,
                           drop_low_confidence: bool = False,
                           confidence_threshold: float = 50.0) -> Tuple[List[int], List[int]]:
        """
        Calculate interface residues between binder and receptor chains.

        Two-stage algorithm:
        1. CB atom prefiltering at cb_cutoff distance
        2. All-atom refinement at all_atom_cutoff distance

        Args:
            cb_cutoff: Distance cutoff for CB prefiltering (Angstroms). Use -1 to skip.
            all_atom_cutoff: Distance cutoff for all-atom refinement (Angstroms). Use -1 to skip.
            min_interface_size: Minimum number of binder residues to form interface
            drop_low_confidence: Whether to filter low-confidence residues after calculation
            confidence_threshold: B-factor/pLDDT threshold for filtering

        Returns:
            Tuple of (binder_interface_residues, receptor_interface_residues)
        """
        if not self.is_ready_for_analysis():
            raise RuntimeError("Complex not ready. Call identify_chains() first.")

        atom_df = self._get_atom_dataframe()
        binder_atoms = self._get_chain_atoms(atom_df, self.binder_chains)
        receptor_atoms = self._get_chain_atoms(atom_df, self.receptor_chains)

        # Stage 1: CB atom prefiltering
        if cb_cutoff != -1:
            binder_cb = self._get_cb_or_ca_atoms(binder_atoms)
            receptor_cb = self._get_cb_or_ca_atoms(receptor_atoms)

            if len(binder_cb) == 0 or len(receptor_cb) == 0:
                print('Binder or receptor CB atoms could not be computed.')
                return [], []

            cb_rec_residues, cb_binder_residues = self._calculate_interface_residues(
                receptor_cb, binder_cb, cb_cutoff
            )

            binder_filtered = binder_atoms[binder_atoms['residue_number'].isin(cb_binder_residues)]
            receptor_filtered = receptor_atoms[receptor_atoms['residue_number'].isin(cb_rec_residues)]
        else:
            binder_filtered = binder_atoms
            receptor_filtered = receptor_atoms

        # Stage 2: All-atom refinement
        if all_atom_cutoff != -1:
            final_binder_res, final_rec_res = self._calculate_interface_residues(
                binder_filtered, receptor_filtered, all_atom_cutoff
            )
        else:
            final_binder_res = cb_binder_residues if cb_cutoff != -1 else []
            final_rec_res = cb_rec_residues if cb_cutoff != -1 else []

        # Filter low-confidence residues AFTER both calculation stages
        if drop_low_confidence:
            binder_interface_atoms = atom_df[
                (atom_df['chain_id'].isin(self.binder_chains)) &
                (atom_df['residue_number'].isin(final_binder_res))
            ]
            high_conf = binder_interface_atoms[binder_interface_atoms['b_factor'] > confidence_threshold]
            final_binder_res = high_conf['residue_number'].unique().tolist()

        # Minimum interface size
        if len(final_binder_res) < min_interface_size:
            final_binder_res, final_rec_res = [], []

        self.interface_residues_binder = final_binder_res
        self.interface_residues_receptor = final_rec_res
        self.interface_calculated = True

        return final_binder_res, final_rec_res

    def _get_atom_dataframe(self) -> pd.DataFrame:
        """Get ATOM DataFrame."""
        if 'ATOM' not in self._df or self._df['ATOM'].empty:
            raise ValueError("No ATOM records found in structure")
        return self._df['ATOM']

    def _get_chain_atoms(self, atom_df: pd.DataFrame, chain_ids: List[str]) -> pd.DataFrame:
        """Get atoms belonging to specified chains."""
        return atom_df[atom_df['chain_id'].isin(chain_ids)]

    def _get_cb_or_ca_atoms(self, atom_df: pd.DataFrame) -> pd.DataFrame:
        """Get CB atoms for all residues, or CA for glycine (excluding hydrogens)."""
        query_str = (
            '((residue_name != "GLY" and atom_name == "CB") or '
            '(residue_name == "GLY" and atom_name == "CA")) and '
            'element_symbol != "H"'
        )
        return atom_df.query(query_str)

    def _calculate_interface_residues(self, atoms1_df: pd.DataFrame,
                                    atoms2_df: pd.DataFrame,
                                    cutoff: float) -> Tuple[List[int], List[int]]:
        """Find interface residues between two atom groups within cutoff distance."""
        coord_cols = ['x_coord', 'y_coord', 'z_coord']
        for df in [atoms1_df, atoms2_df]:
            for col in coord_cols:
                df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')

        distances = atoms2_df.apply(
            lambda row: PandasPdb.distance_df(
                atoms1_df,
                xyz=(row['x_coord'], row['y_coord'], row['z_coord'])
            ),
            axis=1
        )

        interface_residues_1 = atoms1_df.loc[
            distances.min(axis=0) <= cutoff, 'residue_number'
        ].unique().tolist()

        interface_residues_2 = atoms2_df.loc[
            distances.min(axis=1) <= cutoff, 'residue_number'
        ].unique().tolist()

        return interface_residues_1, interface_residues_2

    # ============================================================================
    # INTERFACE ANALYSIS
    # ============================================================================

    def get_interface_atoms(self, atom_type: str = 'all') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get interface atoms for binder and receptor.

        Args:
            atom_type: 'all', 'backbone', 'sidechain', or 'ca'

        Returns:
            Tuple of (binder_interface_atoms, receptor_interface_atoms)
        """
        if not self.interface_calculated:
            raise RuntimeError("Interface not calculated. Call calculate_interface() first.")

        atom_df = self._df['ATOM']

        binder_atoms = atom_df[
            (atom_df['chain_id'].isin(self.binder_chains)) &
            (atom_df['residue_number'].isin(self.interface_residues_binder))
        ]
        receptor_atoms = atom_df[
            (atom_df['chain_id'].isin(self.receptor_chains)) &
            (atom_df['residue_number'].isin(self.interface_residues_receptor))
        ]

        if atom_type != 'all':
            binder_atoms = self._filter_by_atom_type(binder_atoms, atom_type)
            receptor_atoms = self._filter_by_atom_type(receptor_atoms, atom_type)

        return binder_atoms, receptor_atoms

    def _filter_by_atom_type(self, atom_df: pd.DataFrame, atom_type: str) -> pd.DataFrame:
        """Filter atoms by type."""
        if atom_type == 'ca':
            return atom_df[atom_df['atom_name'] == 'CA']
        elif atom_type == 'backbone':
            return atom_df[atom_df['atom_name'].isin(['N', 'CA', 'C', 'O'])]
        elif atom_type == 'sidechain':
            return atom_df[~atom_df['atom_name'].isin(['N', 'CA', 'C', 'O'])]
        return atom_df

    def get_interface_summary(self) -> dict:
        """Get summary statistics of the interface."""
        if not self.interface_calculated:
            return {'status': 'not_calculated'}

        binder_atoms, receptor_atoms = self.get_interface_atoms()
        return {
            'status': 'calculated',
            'binder_interface_residues': len(self.interface_residues_binder),
            'receptor_interface_residues': len(self.interface_residues_receptor),
            'binder_interface_atoms': len(binder_atoms),
            'receptor_interface_atoms': len(receptor_atoms),
            'binder_residue_list': self.interface_residues_binder,
            'receptor_residue_list': self.interface_residues_receptor
        }

    def clear_interface_results(self):
        """Clear interface calculation results."""
        self.interface_residues_binder = None
        self.interface_residues_receptor = None
        self.interface_calculated = False

    def clear_cache(self) -> None:
        """Clear all cached calculations."""
        self._cache = {key: None for key in CACHE_KEYS.values()}
        self._cache['chain_lengths'] = {}
        self.clear_interface_results()
        self.clear_confidence_metrics()

    # ============================================================================
    # CONFIDENCE DATA
    # ============================================================================

    def load_confidence_data(self, confidence_file_path: Optional[str] = None,
                           require_confidence: bool = False) -> bool:
        """
        Load confidence data for the structure.

        Args:
            confidence_file_path: Explicit path to confidence file (optional)
            require_confidence: Whether to raise error if confidence data not found

        Returns:
            True if confidence data loaded successfully
        """
        from ..io.confidence import load_confidence_data

        self.confidence_data = None
        self.has_confidence_data = False

        try:
            result = load_confidence_data(
                self._structure_file_path,
                confidence_file=confidence_file_path
            )

            if result is None:
                if require_confidence:
                    raise RuntimeError("No confidence data found")
                return False

            self.confidence_data = result
            self.has_confidence_data = result.get('has_confidence_data', False)
            return True

        except Exception as e:
            if require_confidence:
                raise RuntimeError(f"Failed to load confidence data: {e}")
            return False

    def get_confidence_summary(self) -> Dict[str, Any]:
        """Get summary of available confidence data."""
        if not self.has_confidence_data:
            return {'status': 'no_confidence_data'}

        summary = {
            'status': 'confidence_loaded',
            'has_pae_matrix': self.confidence_data.get('pae_matrix') is not None,
            'has_iptm': self.confidence_data.get('iptm') is not None,
            'has_ptm': self.confidence_data.get('ptm') is not None,
        }

        if self.confidence_data.get('iptm') is not None:
            summary['iptm'] = round(self.confidence_data['iptm'], 3)
        if self.confidence_data.get('ptm') is not None:
            summary['ptm'] = round(self.confidence_data['ptm'], 3)
        if self.confidence_data.get('confidence') is not None:
            summary['combined_confidence'] = round(self.confidence_data['confidence'], 3)

        return summary

    def calculate_confidence_metrics(self,
                                     require_interface: bool = True) -> Dict[str, Any]:
        """
        Calculate confidence metrics for the interface.

        Returns:
            Dict with pLDDT, PAE, iPTM, pTM metrics
        """
        if require_interface and not self.interface_calculated:
            raise RuntimeError("Interface not calculated. Call calculate_interface() first.")

        if not self.has_confidence_data:
            raise RuntimeError("Confidence data not loaded. Call load_confidence_data() first.")

        metrics = {
            'status': 'calculated',
            'avg_plddt_interface': 0.0,
            'max_plddt_interface': 0.0,
            'interface_pae': 30.0,
            'min_interface_pae': 30.0,
            'iptm': self.confidence_data.get('iptm', 0.0),
            'ptm': self.confidence_data.get('ptm', 0.0),
            'combined_confidence': self.confidence_data.get('confidence', 0.0),
            'has_interface_metrics': False
        }

        if self.interface_calculated and self.interface_residues_binder:
            metrics.update(self._calculate_interface_plddt())
            metrics.update(self._calculate_interface_pae())
            metrics['has_interface_metrics'] = True

        self.confidence_metrics = metrics
        return metrics

    def _calculate_interface_plddt(self) -> Dict[str, float]:
        """Calculate pLDDT metrics for binder interface residues."""
        if not self.interface_residues_binder:
            return {'avg_plddt_interface': 0.0, 'max_plddt_interface': 0.0}

        binder_interface_ca = self._df['ATOM'][
            (self._df['ATOM']['chain_id'].isin(self.binder_chains)) &
            (self._df['ATOM']['residue_number'].isin(self.interface_residues_binder)) &
            (self._df['ATOM']['atom_name'] == 'CA')
        ].sort_values('residue_number')

        if binder_interface_ca.empty:
            return {'avg_plddt_interface': 0.0, 'max_plddt_interface': 0.0}

        plddt_values = binder_interface_ca['b_factor']
        return {
            'avg_plddt_interface': round(float(plddt_values.mean()), 2),
            'max_plddt_interface': round(float(plddt_values.max()), 2)
        }

    def _calculate_interface_pae(self) -> Dict[str, float]:
        """Calculate PAE metrics for interface residues."""
        default_pae = {'interface_pae': 30.0, 'min_interface_pae': 30.0}

        if not self.confidence_data or self.confidence_data.get('pae_matrix') is None:
            return default_pae

        pae_matrix = self.confidence_data['pae_matrix']

        try:
            binder_ca = self._df['ATOM'][
                (self._df['ATOM']['chain_id'].isin(self.binder_chains)) &
                (self._df['ATOM']['residue_number'].isin(self.interface_residues_binder)) &
                (self._df['ATOM']['atom_name'] == 'CA')
            ]
            receptor_ca = self._df['ATOM'][
                (self._df['ATOM']['chain_id'].isin(self.receptor_chains)) &
                (self._df['ATOM']['residue_number'].isin(self.interface_residues_receptor)) &
                (self._df['ATOM']['atom_name'] == 'CA')
            ]

            if binder_ca.empty or receptor_ca.empty:
                return default_pae

            binder_indices = binder_ca['ca_index'].tolist()
            receptor_indices = receptor_ca['ca_index'].tolist()

            if binder_indices and receptor_indices:
                submatrix = pae_matrix.iloc[binder_indices, receptor_indices]
                return {
                    'interface_pae': round(float(submatrix.median().median()), 2),
                    'min_interface_pae': round(float(submatrix.min().min()), 2)
                }

            return default_pae

        except Exception as e:
            print(f"Error calculating interface PAE: {e}")
            return default_pae

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all calculated metrics."""
        summary = {
            'structure_info': self.get_analysis_summary(),
            'interface_info': self.get_interface_summary(),
            'confidence_info': self.get_confidence_summary()
        }
        if hasattr(self, 'confidence_metrics'):
            summary['metrics'] = self.confidence_metrics
        else:
            summary['metrics'] = {'status': 'not_calculated'}
        return summary

    def calculate_all_metrics(self,
                              cb_cutoff: float = 8.0,
                              all_atom_cutoff: float = 4.0,
                              min_interface_size: int = 1,
                              drop_low_confidence: bool = False,
                              confidence_threshold: float = 50.0,
                              require_confidence: bool = False) -> Dict[str, Any]:
        """
        Complete analysis workflow: interface + confidence + metrics.

        Args:
            cb_cutoff: CB atom distance cutoff
            all_atom_cutoff: All-atom distance cutoff
            min_interface_size: Minimum interface size
            drop_low_confidence: Filter low confidence residues
            confidence_threshold: Confidence threshold for filtering
            require_confidence: Whether to require confidence data

        Returns:
            Dict with complete analysis results
        """
        results = {}

        if not self.interface_calculated:
            binder_int, rec_int = self.calculate_interface(
                cb_cutoff=cb_cutoff,
                all_atom_cutoff=all_atom_cutoff,
                min_interface_size=min_interface_size,
                drop_low_confidence=drop_low_confidence,
                confidence_threshold=confidence_threshold
            )
            results['interface_residues'] = {
                'binder': binder_int,
                'receptor': rec_int
            }

        if not self.has_confidence_data:
            loaded = self.load_confidence_data(require_confidence=require_confidence)
            results['confidence_loaded'] = loaded
            if not loaded and require_confidence:
                raise RuntimeError("Confidence data required but could not be loaded")

        if self.has_confidence_data:
            results['metrics'] = self.calculate_confidence_metrics(require_interface=True)
        else:
            results['metrics'] = {'status': 'no_confidence_data'}

        results['summary'] = self.get_metrics_summary()
        return results

    def clear_confidence_metrics(self):
        """Clear calculated confidence metrics."""
        if hasattr(self, 'confidence_metrics'):
            delattr(self, 'confidence_metrics')

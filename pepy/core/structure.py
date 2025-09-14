"""
Main peptide-protein complex structure class.
"""

import warnings
import os
import math
import numpy as np
from typing import Optional, List, Dict, Union, Tuple, Any

import pandas as pd
from biopandas.pdb import PandasPdb

from ..io.loaders import load_pdb_file, load_cif_file, fix_af3_file_safe, add_ca_indices
from ..core.base import DEFAULT_CB_CUTOFF, DEFAULT_ALL_ATOM_CUTOFF, CACHE_KEYS
from ..utils.validation import validate_file_path, validate_chains, validate_analysis_ready

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class PeptideProteinComplex(PandasPdb):
    """
    Extended biopandas structure class for peptide-protein complex analysis.

    Handles single peptide complexes with one or more receptor chains.
    Provides caching for expensive calculations and systematic chain identification.
    """

    def __init__(self):
        super().__init__()

        # Chain assignments
        self.peptide_chain: Optional[str] = None
        self.receptor_chains: List[str] = []

        # Cache for expensive calculations
        self._cache = {key: None for key in CACHE_KEYS.values()}
        self._cache['chain_lengths'] = {}

        # Interface results
        self.interface_residues_peptide: Optional[List[int]] = None
        self.interface_residues_receptor: Optional[List[int]] = None
        self.interface_calculated: bool = False

        # Confidence data
        self.confidence_data: Optional[Dict[str, Any]] = None
        self.has_confidence_data: bool = False
        self._structure_file_path: Optional[str] = None

        # Analysis parameters
        self.cb_cutoff = DEFAULT_CB_CUTOFF
        self.all_atom_cutoff = DEFAULT_ALL_ATOM_CUTOFF

    @classmethod
    def from_file(cls, filepath: str, **kwargs) -> 'PeptideProteinComplex':
        """
        Load structure from PDB or CIF file.

        Args:
            filepath: Path to structure file (.pdb or .cif)
            **kwargs: Additional arguments for loading

        Returns:
            PeptideProteinComplex instance
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

        # Store file path for confidence file discovery
        self._structure_file_path = os.path.abspath(filepath)

        if filepath.endswith('.cif'):
            if fix_af3:
                fix_af3_file_safe(filepath)
            self._df = load_cif_file(filepath)
        elif filepath.endswith('.pdb'):
            self._df = load_pdb_file(filepath)

        # Add CA indices for PAE matrix mapping
        if 'ATOM' in self._df and not self._df['ATOM'].empty:
            self._df['ATOM'] = add_ca_indices(self._df['ATOM'])

        # Clear cache when loading new structure
        self.clear_cache()

    def identify_chains(self,
                       peptide_chain: Optional[str] = None,
                       receptor_chains: Optional[List[str]] = None) -> None:
        """
        Identify peptide and receptor chains in the complex.

        Logic:
        1. If peptide_chain provided: use it, rest are receptors (or specified receptors)
        2. If only receptor_chains provided: remaining single chain becomes peptide
        3. If nothing specified: shortest chain becomes peptide

        Args:
            peptide_chain: Single peptide chain ID
            receptor_chains: List of receptor chain IDs

        Raises:
            ValueError: If chain identification logic fails
        """
        available_chains = self.get_available_chains()
        validate_chains(available_chains, peptide_chain, receptor_chains)

        if peptide_chain:
            self.peptide_chain = peptide_chain

            if receptor_chains:
                self.receptor_chains = receptor_chains
            else:
                # All other chains are receptors
                self.receptor_chains = [c for c in available_chains if c != peptide_chain]

        elif receptor_chains:
            # Only receptor chains specified
            remaining_chains = [c for c in available_chains if c not in receptor_chains]

            if len(remaining_chains) == 1:
                self.peptide_chain = remaining_chains[0]
                self.receptor_chains = receptor_chains
            elif len(remaining_chains) == 0:
                raise ValueError("No chains left for peptide after specifying receptors")
            else:
                raise ValueError(
                    f"Multiple non-receptor chains found: {remaining_chains}. "
                    "Please specify peptide_chain explicitly."
                )

        else:
            # Auto-detect: shortest chain is peptide
            chain_lengths = self.get_chain_lengths()
            shortest_chain = min(chain_lengths.items(), key=lambda x: x[1])[0]

            self.peptide_chain = shortest_chain
            self.receptor_chains = [c for c in available_chains if c != shortest_chain]

        # Clear cache after chain assignment
        self.clear_cache()

        print(f'Peptide chain: {self.peptide_chain}, receptor chains: {",".join(self.receptor_chains)}')

    def get_available_chains(self) -> List[str]:
        """Get list of available chain IDs in the structure."""
        if 'ATOM' not in self._df or self._df['ATOM'].empty:
            return []
        return sorted(self._df['ATOM']['chain_id'].unique().tolist())

    def get_chain_lengths(self) -> Dict[str, int]:
        """
        Get number of residues for each chain.

        Returns:
            Dict mapping chain_id to residue count
        """
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
        """
        Get comprehensive information about all chains.

        Returns:
            Dict with chain info including length, type, residue range
        """
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
        """Determine if chain is peptide, receptor, or unassigned."""
        if chain_id == self.peptide_chain:
            return 'peptide'
        elif chain_id in self.receptor_chains:
            return 'receptor'
        else:
            return 'unassigned'

    def _get_chain_sequence(self, chain_id: str) -> str:
        """Get single-letter amino acid sequence for chain."""
        # Create a temporary PandasPdb with just this chain's data
        chain_atoms = self._df['ATOM'][self._df['ATOM']['chain_id'] == chain_id]

        if chain_atoms.empty:
            return ""

        # Get CA atoms sorted by residue number
        ca_atoms = chain_atoms[chain_atoms['atom_name'] == 'CA'].sort_values('residue_number')

        if ca_atoms.empty:
            return ""

        # Create temporary PandasPdb object for amino3to1 conversion
        temp_pdb = PandasPdb()
        temp_pdb._df = {'ATOM': ca_atoms, 'HETATM': self._df.get('HETATM', ca_atoms.iloc[:0].copy())}

        seq_df = temp_pdb.amino3to1()
        # Filter for this chain and sort by residue number
        chain_seq = seq_df[seq_df['chain_id'] == chain_id]
        sequence = ''.join(chain_seq['residue_name'].values)
        return sequence

    def is_ready_for_analysis(self) -> bool:
        """Check if structure is ready for peptide-protein analysis."""
        try:
            validate_analysis_ready(
                self.peptide_chain,
                self.receptor_chains,
                'ATOM' not in self._df or self._df['ATOM'].empty
            )
            return True
        except RuntimeError:
            return False

    def get_analysis_summary(self) -> Dict:
        """Get summary of current structure and chain assignments."""
        if not self.is_ready_for_analysis():
            return {'status': 'not_ready', 'message': 'Chain identification required'}

        chain_info = self.get_chain_info()

        return {
            'status': 'ready',
            'peptide_chain': self.peptide_chain,
            'receptor_chains': self.receptor_chains,
            'peptide_length': chain_info[self.peptide_chain]['length'],
            'total_receptor_length': sum(
                chain_info[c]['length'] for c in self.receptor_chains
            ),
            'total_chains': len(self.get_available_chains()),
            'unassigned_chains': [
                c for c in self.get_available_chains()
                if c not in [self.peptide_chain] + self.receptor_chains
            ]
        }

    # ============================================================================
    # INTERFACE CALCULATION METHODS (Integrated from InterfaceCalculator)
    # ============================================================================

    def calculate_interface(self,
                           cb_cutoff: float = 8.0,
                           all_atom_cutoff: float = 4.0,
                           min_interface_size: int = 1,
                           drop_low_confidence: bool = False,
                           confidence_threshold: float = 50.0) -> Tuple[List[int], List[int]]:
        """
        Calculate interface residues between peptide and receptor chains.

        Args:
            cb_cutoff: Distance cutoff for CB atom prefiltering (Angstroms)
            all_atom_cutoff: Distance cutoff for all-atom refinement (Angstroms)
            min_interface_size: Minimum number of residues to form interface
            drop_low_confidence: Whether to filter low confidence residues
            confidence_threshold: B-factor/pLDDT threshold for filtering

        Returns:
            Tuple of (peptide_interface_residues, receptor_interface_residues)
        """
        # Validate that chains are identified
        if not self.is_ready_for_analysis():
            raise RuntimeError("Complex not ready for analysis. Call identify_chains() first.")

        # Get atom dataframes for peptide and receptor chains
        atom_df = self._get_atom_dataframe()
        peptide_atoms = self._get_chain_atoms(atom_df, [self.peptide_chain])
        receptor_atoms = self._get_chain_atoms(atom_df, self.receptor_chains)

        # Stage 1: CB atom prefiltering (if cb_cutoff != -1)
        if cb_cutoff != -1:
            peptide_cb_atoms = self._get_cb_or_ca_atoms(peptide_atoms)
            receptor_cb_atoms = self._get_cb_or_ca_atoms(receptor_atoms)

            if len(peptide_cb_atoms) == 0 or len(receptor_cb_atoms) == 0:
                print('Peptide or receptor CB atoms could not be computed.')
                return [], []

            cb_rec_residues, cb_pep_residues = self._calculate_interface_residues(
                receptor_cb_atoms, peptide_cb_atoms, cb_cutoff
            )

            # Filter original atoms to CB-identified interface regions
            peptide_filtered = peptide_atoms[
                peptide_atoms['residue_number'].isin(cb_pep_residues)
            ]
            receptor_filtered = receptor_atoms[
                receptor_atoms['residue_number'].isin(cb_rec_residues)
            ]
        else:
            # No CB prefiltering - use all atoms
            peptide_filtered = peptide_atoms
            receptor_filtered = receptor_atoms

        # Filter peptide for low confidence
        if drop_low_confidence:
            peptide_filtered = self._filter_by_confidence_threshold(
                peptide_filtered, confidence_threshold
            )

        # Stage 2: All-atom refinement (if all_atom_cutoff != -1)
        if all_atom_cutoff != -1:
            final_pep_residues, final_rec_residues = self._calculate_interface_residues(
                peptide_filtered, receptor_filtered, all_atom_cutoff
            )
        else:
            # No all-atom refinement - use CB results
            final_pep_residues = cb_pep_residues if cb_cutoff != -1 else []
            final_rec_residues = cb_rec_residues if cb_cutoff != -1 else []

        # Apply minimum interface size filtering
        final_pep_residues = self._filter_by_minimum_size(
            final_pep_residues, min_interface_size
        )

        # If there are no final_pep_residues, also remove final_rec_residues
        if len(final_pep_residues) == 0:
            final_pep_residues, final_rec_residues = [], []

        # Store results directly on the complex
        self.interface_residues_peptide = final_pep_residues
        self.interface_residues_receptor = final_rec_residues
        self.interface_calculated = True

        return final_pep_residues, final_rec_residues

    def _get_atom_dataframe(self) -> pd.DataFrame:
        """Get ATOM DataFrame from the complex."""
        if 'ATOM' not in self._df or self._df['ATOM'].empty:
            raise ValueError("No ATOM records found in structure")
        return self._df['ATOM']

    def _get_chain_atoms(self, atom_df: pd.DataFrame, chain_ids: List[str]) -> pd.DataFrame:
        """Get atoms belonging to specified chains."""
        return atom_df[atom_df['chain_id'].isin(chain_ids)]

    def _get_cb_or_ca_atoms(self, atom_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get CB atoms for all residues, or CA for glycine.

        Args:
            atom_df: DataFrame with ATOM records

        Returns:
            DataFrame with CB/CA atoms only
        """
        # CB atoms for non-glycine, CA for glycine, exclude hydrogens
        query_str = (
            '((residue_name != "GLY" and atom_name == "CB") or '
            '(residue_name == "GLY" and atom_name == "CA")) and '
            'element_symbol != "H"'
        )
        return atom_df.query(query_str)

    def _calculate_interface_residues(self, atoms1_df: pd.DataFrame,
                                    atoms2_df: pd.DataFrame,
                                    cutoff: float) -> Tuple[List[int], List[int]]:
        """
        Find interface residues between two atom groups within cutoff distance.

        Args:
            atoms1_df: First group of atoms
            atoms2_df: Second group of atoms
            cutoff: Distance cutoff in Angstroms

        Returns:
            Tuple of (group1_interface_residues, group2_interface_residues)
        """
        # Ensure coordinates are float type
        coord_cols = ['x_coord', 'y_coord', 'z_coord']
        for df in [atoms1_df, atoms2_df]:
            for col in coord_cols:
                df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate distances using biopandas method for each atom in group2
        distances = atoms2_df.apply(
            lambda row: PandasPdb.distance_df(
                atoms1_df,
                xyz=(row['x_coord'], row['y_coord'], row['z_coord'])
            ),
            axis=1
        )

        # Find residues within cutoff
        interface_residues_1 = atoms1_df.loc[
            distances.min(axis=0) <= cutoff, 'residue_number'
        ].unique().tolist()

        interface_residues_2 = atoms2_df.loc[
            distances.min(axis=1) <= cutoff, 'residue_number'
        ].unique().tolist()

        return interface_residues_1, interface_residues_2

    def _filter_by_confidence_threshold(self, interface_atoms: pd.DataFrame,
                                      threshold: float) -> pd.DataFrame:
        """Filter interface atoms by confidence threshold."""
        return interface_atoms[interface_atoms['b_factor'] > threshold]

    def _filter_by_minimum_size(self, interface_residues: List[int],
                               min_size: int = 1) -> List[int]:
        """
        Filter interface by minimum number of residues.

        Args:
            interface_residues: List of interface residue numbers
            min_size: Minimum number of residues required

        Returns:
            Original list if size >= min_size, empty list otherwise
        """
        if len(interface_residues) >= min_size:
            return interface_residues
        return []

    # ============================================================================
    # INTERFACE ANALYSIS METHODS (Integrated from InterfaceAnalyzer)
    # ============================================================================

    def get_interface_atoms(self, atom_type: str = 'all') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get interface atoms for peptide and receptor.

        Args:
            atom_type: 'all', 'backbone', 'sidechain', or 'ca'

        Returns:
            Tuple of (peptide_interface_atoms, receptor_interface_atoms)
        """
        if not self.interface_calculated:
            raise RuntimeError("Interface not calculated. Call calculate_interface() first.")

        atom_df = self._df['ATOM']

        # Get interface atoms
        peptide_interface_atoms = atom_df[
            (atom_df['chain_id'] == self.peptide_chain) &
            (atom_df['residue_number'].isin(self.interface_residues_peptide))
        ]

        receptor_interface_atoms = atom_df[
            (atom_df['chain_id'].isin(self.receptor_chains)) &
            (atom_df['residue_number'].isin(self.interface_residues_receptor))
        ]

        # Filter by atom type if requested
        if atom_type != 'all':
            peptide_interface_atoms = self._filter_by_atom_type(peptide_interface_atoms, atom_type)
            receptor_interface_atoms = self._filter_by_atom_type(receptor_interface_atoms, atom_type)

        return peptide_interface_atoms, receptor_interface_atoms

    def _filter_by_atom_type(self, atom_df: pd.DataFrame, atom_type: str) -> pd.DataFrame:
        """Filter atoms by type."""
        if atom_type == 'ca':
            return atom_df[atom_df['atom_name'] == 'CA']
        elif atom_type == 'backbone':
            return atom_df[atom_df['atom_name'].isin(['N', 'CA', 'C', 'O'])]
        elif atom_type == 'sidechain':
            return atom_df[~atom_df['atom_name'].isin(['N', 'CA', 'C', 'O'])]
        else:
            return atom_df

    def get_interface_summary(self) -> dict:
        """Get summary statistics of the interface."""
        if not self.interface_calculated:
            return {'status': 'not_calculated'}

        peptide_atoms, receptor_atoms = self.get_interface_atoms()

        return {
            'status': 'calculated',
            'peptide_interface_residues': len(self.interface_residues_peptide),
            'receptor_interface_residues': len(self.interface_residues_receptor),
            'peptide_interface_atoms': len(peptide_atoms),
            'receptor_interface_atoms': len(receptor_atoms),
            'peptide_residue_list': self.interface_residues_peptide,
            'receptor_residue_list': self.interface_residues_receptor
        }

    def clear_interface_results(self):
        """Clear interface calculation results."""
        self.interface_residues_peptide = None
        self.interface_residues_receptor = None
        self.interface_calculated = False

    def clear_cache(self) -> None:
        """Clear all cached calculations."""
        self._cache = {key: None for key in CACHE_KEYS.values()}
        self._cache['chain_lengths'] = {}
        # Clear interface results
        self.clear_interface_results()
        # Clear confidence metrics
        self.clear_confidence_metrics()

    # ============================================================================
    # CONFIDENCE DATA METHODS
    # ============================================================================

    def load_confidence_data(self, confidence_file_path: Optional[str] = None,
                           require_confidence: bool = False) -> bool:
        """
        Load confidence data for the structure.

        Args:
            confidence_file_path: Explicit path to confidence file (optional)
            require_confidence: Whether to raise error if confidence data not found

        Returns:
            True if confidence data loaded successfully, False otherwise

        Raises:
            RuntimeError: If require_confidence=True and files not found
        """
        from ..io.confidence import ConfidenceFileFinder, ConfidenceFileLoader
        from ..io.parsers import ConfidenceMetricsParser

        # Clear any existing confidence data
        self.confidence_data = None
        self.has_confidence_data = False

        try:
            if confidence_file_path:
                # Use explicit file path
                file_info = {'type': 'manual', 'confidence_file': confidence_file_path}
                raw_data = ConfidenceFileLoader.load_confidence_data(file_info)
            else:
                # Auto-discover confidence files
                finder = ConfidenceFileFinder()
                if not hasattr(self, '_structure_file_path'):
                    if require_confidence:
                        raise RuntimeError("Structure file path not tracked. Cannot auto-discover confidence files.")
                    return False

                file_info = finder.find_confidence_files(self._structure_file_path, require_confidence)
                if not file_info:
                    return False

                raw_data = ConfidenceFileLoader.load_confidence_data(file_info)

            # Parse and store confidence data
            self.confidence_data = ConfidenceMetricsParser.parse_confidence_data(raw_data)
            self.has_confidence_data = self.confidence_data['has_confidence_data']

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
            'has_pae_matrix': self.confidence_data['pae_matrix'] is not None,
            'has_iptm': self.confidence_data['iptm'] is not None,
            'has_ptm': self.confidence_data['ptm'] is not None,
            'has_plddt': self.confidence_data['plddt'] is not None,
            'combined_confidence': self.confidence_data['confidence']
        }

        if self.confidence_data['iptm'] is not None:
            summary['iptm'] = round(self.confidence_data['iptm'], 3)
        if self.confidence_data['ptm'] is not None:
            summary['ptm'] = round(self.confidence_data['ptm'], 3)
        if self.confidence_data['confidence'] is not None:
            summary['combined_confidence'] = round(self.confidence_data['confidence'], 3)

        return summary


    def calculate_confidence_metrics(self,
                                     window_size: Optional[int] = None,
                                     require_interface: bool = True) -> Dict[str, Any]:
        """
        Calculate confidence metrics for the interface.

        Args:
            window_size: Window size for pLDDT calculation (None = no windowing, exact interface)
            require_interface: Whether to require interface calculation first

        Returns:
            Dict with calculated confidence metrics

        Raises:
            RuntimeError: If interface not calculated or confidence data not loaded
        """
        # Validate prerequisites
        if require_interface and not self.interface_calculated:
            raise RuntimeError("Interface not calculated. Call calculate_interface() first.")

        if not self.has_confidence_data:
            raise RuntimeError("Confidence data not loaded. Call load_confidence_data() first.")

        # Initialize results
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

        # Calculate interface-specific metrics if interface exists
        if self.interface_calculated and len(self.interface_residues_peptide) > 0:
            # Calculate interface pLDDT metrics
            interface_plddt_metrics = self._calculate_interface_plddt(window_size)
            metrics.update(interface_plddt_metrics)

            # Calculate interface PAE metrics
            interface_pae_metrics = self._calculate_interface_pae()
            metrics.update(interface_pae_metrics)

            metrics['has_interface_metrics'] = True

        # Store metrics on the complex
        self.confidence_metrics = metrics

        return metrics


    def _calculate_interface_plddt(self, window_size: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate pLDDT metrics for interface residues.

        Args:
            window_size: Window size for averaging (None = exact interface only)

        Returns:
            Dict with pLDDT metrics
        """
        if not self.interface_residues_peptide:
            return {'avg_plddt_interface': 0.0, 'max_plddt_interface': 0.0}

        # Get interface CA atoms for peptide chain
        peptide_interface_atoms = self._df['ATOM'][
            (self._df['ATOM']['chain_id'] == self.peptide_chain) &
            (self._df['ATOM']['residue_number'].isin(self.interface_residues_peptide)) &
            (self._df['ATOM']['atom_name'] == 'CA')
            ].sort_values('residue_number')

        if peptide_interface_atoms.empty:
            return {'avg_plddt_interface': 0.0, 'max_plddt_interface': 0.0}

        # Extract B-factors (pLDDT values)
        plddt_values = peptide_interface_atoms['b_factor']

        if window_size is None or window_size >= len(plddt_values):
            # Use exact interface residues (no windowing)
            avg_plddt = float(plddt_values.mean())
            max_plddt = float(plddt_values.max())
        else:
            # Apply windowing (rolling average)
            windowed_plddt = plddt_values.rolling(window=window_size, center=True).mean()
            # Get maximum of windowed averages, excluding NaN values
            valid_windowed = windowed_plddt.dropna()
            if not valid_windowed.empty:
                avg_plddt = float(valid_windowed.max())
                max_plddt = float(plddt_values.max())
            else:
                avg_plddt = float(plddt_values.mean())
                max_plddt = float(plddt_values.max())

        return {
            'avg_plddt_interface': round(avg_plddt, 2),
            'max_plddt_interface': round(max_plddt, 2)
        }


    def _calculate_interface_pae(self) -> Dict[str, float]:
        """
        Calculate PAE metrics for interface residues.

        Returns:
            Dict with PAE metrics
        """
        # Default values if calculation fails
        default_pae = {'interface_pae': 30.0, 'min_interface_pae': 30.0}

        # Check if we have PAE matrix
        if not self.confidence_data or self.confidence_data.get('pae_matrix') is None:
            return default_pae

        pae_matrix = self.confidence_data['pae_matrix']

        # Get CA indices for interface residues
        try:
            # Get peptide interface CA indices
            peptide_interface_atoms = self._df['ATOM'][
                (self._df['ATOM']['chain_id'] == self.peptide_chain) &
                (self._df['ATOM']['residue_number'].isin(self.interface_residues_peptide)) &
                (self._df['ATOM']['atom_name'] == 'CA')
                ]

            # Get receptor interface CA indices
            receptor_interface_atoms = self._df['ATOM'][
                (self._df['ATOM']['chain_id'].isin(self.receptor_chains)) &
                (self._df['ATOM']['residue_number'].isin(self.interface_residues_receptor)) &
                (self._df['ATOM']['atom_name'] == 'CA')
                ]

            if peptide_interface_atoms.empty or receptor_interface_atoms.empty:
                return default_pae

            # Extract CA indices
            pep_ca_indices = peptide_interface_atoms['ca_index'].tolist()
            rec_ca_indices = receptor_interface_atoms['ca_index'].tolist()

            # Extract interface PAE values (vectorized approach)
            if pep_ca_indices and rec_ca_indices:
                interface_pae_submatrix = pae_matrix.iloc[pep_ca_indices, rec_ca_indices]
                interface_pae = round(interface_pae_submatrix.median().median(), 2)
                min_interface_pae = round(interface_pae_submatrix.min().min(), 2)
                return {
                    'interface_pae': interface_pae,
                    'min_interface_pae': min_interface_pae
                }
            else:
                return default_pae

        except Exception as e:
            print(f"Error calculating interface PAE: {e}")
            return default_pae


    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of all calculated metrics.

        Returns:
            Dict with all available metrics
        """
        summary = {
            'structure_info': self.get_analysis_summary(),
            'interface_info': self.get_interface_summary(),
            'confidence_info': self.get_confidence_summary()
        }

        # Add calculated metrics if available
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
                              window_size: Optional[int] = None,
                              require_confidence: bool = False) -> Dict[str, Any]:
        """
        Complete analysis workflow: interface + confidence + metrics.

        Args:
            cb_cutoff: CB atom distance cutoff
            all_atom_cutoff: All-atom distance cutoff
            min_interface_size: Minimum interface size
            drop_low_confidence: Filter low confidence residues
            confidence_threshold: Confidence threshold for filtering
            window_size: Window size for pLDDT (None = exact interface)
            require_confidence: Whether to require confidence data

        Returns:
            Dict with complete analysis results
        """
        results = {}

        # Step 1: Calculate interface
        if not self.interface_calculated:
            pep_interface, rec_interface = self.calculate_interface(
                cb_cutoff=cb_cutoff,
                all_atom_cutoff=all_atom_cutoff,
                min_interface_size=min_interface_size,
                drop_low_confidence=drop_low_confidence,
                confidence_threshold=confidence_threshold
            )
            results['interface_residues'] = {
                'peptide': pep_interface,
                'receptor': rec_interface
            }

        # Step 2: Load confidence data if not already loaded
        if not self.has_confidence_data:
            confidence_loaded = self.load_confidence_data(require_confidence=require_confidence)
            results['confidence_loaded'] = confidence_loaded

            if not confidence_loaded and require_confidence:
                raise RuntimeError("Confidence data required but could not be loaded")

        # Step 3: Calculate metrics if confidence data available
        if self.has_confidence_data:
            metrics = self.calculate_confidence_metrics(
                window_size=window_size,
                require_interface=True
            )
            results['metrics'] = metrics
        else:
            results['metrics'] = {'status': 'no_confidence_data'}

        # Step 4: Get comprehensive summary
        results['summary'] = self.get_metrics_summary()

        return results


    # Add clear method for metrics
    def clear_confidence_metrics(self):
        """Clear calculated confidence metrics."""
        if hasattr(self, 'confidence_metrics'):
            delattr(self, 'confidence_metrics')
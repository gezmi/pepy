"""
Main peptide-protein complex structure class.
"""

import warnings
import os
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

        print(f'Peptide chain: {self.peptide_chain}, receptor chains: {','.join(self.receptor_chains)}')

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

    def clear_cache(self) -> None:
        """Clear all cached calculations."""
        self._cache = {key: None for key in CACHE_KEYS.values()}
        self._cache['chain_lengths'] = {}

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
        from ..core.interface import InterfaceCalculator

        calculator = InterfaceCalculator(self)
        pep_residues, rec_residues = calculator.calculate(
            cb_cutoff=cb_cutoff,
            all_atom_cutoff=all_atom_cutoff,
            min_interface_size=min_interface_size,
            drop_low_confidence=drop_low_confidence,
            confidence_threshold=confidence_threshold
        )

        # Store results directly on the complex
        self.interface_residues_peptide = pep_residues
        self.interface_residues_receptor = rec_residues
        self.interface_calculated = True

        return pep_residues, rec_residues

    def get_interface_summary(self) -> dict:
        """Get summary of interface calculation results."""
        from ..core.interface import InterfaceAnalyzer
        analyzer = InterfaceAnalyzer(self)
        return analyzer.get_interface_summary()

    def get_interface_atoms(self, atom_type: str = 'all') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get interface atoms for peptide and receptor."""
        from ..core.interface import InterfaceAnalyzer
        analyzer = InterfaceAnalyzer(self)
        return analyzer.get_interface_atoms(atom_type)

    def clear_interface_results(self):
        """Clear interface calculation results."""
        self.interface_residues_peptide = None
        self.interface_residues_receptor = None
        self.interface_calculated = False

    def clear_cache(self) -> None:
        """Clear all cached calculations."""
        self._cache = {key: None for key in CACHE_KEYS.values()}
        self._cache['chain_lengths'] = {}
        # Clear interface results too
        self.clear_interface_results()

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
                # Get structure file path (need to track this in load_structure)
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
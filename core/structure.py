"""
Main peptide-protein complex structure class.
"""

import warnings
from typing import Optional, List, Dict, Union

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

		if filepath.endswith('.cif'):
			if fix_af3:
				fix_af3_file_safe(filepath)
			self.df = load_cif_file(filepath)
		elif filepath.endswith('.pdb'):
			self.df = load_pdb_file(filepath)

		# Add CA indices for PAE matrix mapping
		if 'ATOM' in self.df and not self.df['ATOM'].empty:
			self.df['ATOM'] = add_ca_indices(self.df['ATOM'])

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

	def get_available_chains(self) -> List[str]:
		"""Get list of available chain IDs in the structure."""
		if 'ATOM' not in self.df or self.df['ATOM'].empty:
			return []
		return sorted(self.df['ATOM']['chain_id'].unique().tolist())

	def get_chain_lengths(self) -> Dict[str, int]:
		"""
		Get number of residues for each chain.

		Returns:
			Dict mapping chain_id to residue count
		"""
		if not self._cache['chain_lengths']:
			if 'ATOM' not in self.df or self.df['ATOM'].empty:
				self._cache['chain_lengths'] = {}
			else:
				self._cache['chain_lengths'] = (
					self.df['ATOM']
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
			chain_atoms = self.df['ATOM'][self.df['ATOM']['chain_id'] == chain_id]

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
		chain_atoms = self.df['ATOM'][self.df['ATOM']['chain_id'] == chain_id]

		# Get unique residues sorted by residue number
		chain_residues = (
			chain_atoms[chain_atoms['atom_name'] == 'CA']
			.drop_duplicates(['residue_number'])
			.sort_values('residue_number')
		)

		# Use biopandas built-in amino acid conversion
		sequence = ''.join([
			self.amino3to1(residue)
			for residue in chain_residues['residue_name']
		])

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
				'ATOM' not in self.df or self.df['ATOM'].empty
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
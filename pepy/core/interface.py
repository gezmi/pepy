"""
Interface calculation logic for peptide-protein complexes.
"""

import pandas as pd
from typing import List, Tuple, Optional, TYPE_CHECKING

from ..utils.geometry import GeometryUtils, InterfaceFilter

if TYPE_CHECKING:
	from .structure import PeptideProteinComplex


class InterfaceCalculator:
	"""
	Calculator for peptide-protein interface detection.

	Implements two-stage calculation:
	1. CB atom prefiltering for speed
	2. All-atom refinement on filtered subset
	"""

	def __init__(self, complex_structure: 'PeptideProteinComplex'):
		"""
		Initialize calculator with structure.

		Args:
			complex_structure: PeptideProteinComplex instance
		"""
		self.complex = complex_structure
		self.geometry = GeometryUtils()
		self.filter = InterfaceFilter()

	def calculate(self,
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
		if not self.complex.is_ready_for_analysis():
			raise RuntimeError("Complex not ready for analysis. Call identify_chains() first.")

		# Get atom dataframes for peptide and receptor chains
		atom_df = self._get_atom_dataframe()
		peptide_atoms = self._get_chain_atoms(atom_df, [self.complex.peptide_chain])
		receptor_atoms = self._get_chain_atoms(atom_df, self.complex.receptor_chains)

		# Stage 1: CB atom prefiltering (if cb_cutoff != -1)
		if cb_cutoff != -1:
			peptide_cb_atoms = self.geometry.get_cb_or_ca_atoms(peptide_atoms)
			receptor_cb_atoms = self.geometry.get_cb_or_ca_atoms(receptor_atoms)

			if len(peptide_cb_atoms) == 0 or len(receptor_cb_atoms) == 0:
				print('Peptide or receptor Cb atoms could not be computed.')
				return [], []

			cb_rec_residues, cb_pep_residues = self.geometry.calculate_interface_residues(
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
			final_pep_residues, final_rec_residues = self.geometry.calculate_interface_residues(
				receptor_filtered, peptide_filtered, all_atom_cutoff
			)
		else:
			# No all-atom refinement - use CB results
			final_pep_residues = cb_pep_residues if cb_cutoff != -1 else []
			final_rec_residues = cb_rec_residues if cb_cutoff != -1 else []

		# Apply minimum interface size filtering
		final_pep_residues = self.filter.filter_by_minimum_size(
			final_pep_residues, min_interface_size
		)

		# If there are no final_pep_residues, also remove final_rec_residues
		if len(final_pep_residues) == 0:
			return [], []
		else:
			return final_pep_residues, final_rec_residues

	def _get_atom_dataframe(self) -> pd.DataFrame:
		"""Get ATOM DataFrame from the complex."""
		if 'ATOM' not in self.complex._df or self.complex._df['ATOM'].empty:
			raise ValueError("No ATOM records found in structure")
		return self.complex._df['ATOM']

	def _get_chain_atoms(self, atom_df: pd.DataFrame, chain_ids: List[str]) -> pd.DataFrame:
		"""Get atoms belonging to specified chains."""
		return atom_df[atom_df['chain_id'].isin(chain_ids)]

	def _filter_by_confidence_threshold(self,
										interface_atoms: pd.DataFrame,
										threshold: float) -> List[int]:

		# Filter by confidence and get unique residues
		high_conf_residues = interface_atoms[
			interface_atoms['b_factor'] > threshold
			]

		return high_conf_residues


class InterfaceAnalyzer:
	"""
	Analysis utilities for interface results.
	"""

	def __init__(self, complex_structure: 'PeptideProteinComplex'):
		self.complex = complex_structure

	def get_interface_atoms(self, atom_type: str = 'all') -> Tuple[pd.DataFrame, pd.DataFrame]:
		"""
		Get interface atoms for peptide and receptor.

		Args:
			atom_type: 'all', 'backbone', 'sidechain', or 'ca'

		Returns:
			Tuple of (peptide_interface_atoms, receptor_interface_atoms)
		"""
		if not self.complex.interface_calculated:
			raise RuntimeError("Interface not calculated. Call calculate_interface() first.")

		atom_df = self.complex._df['ATOM']

		# Get interface atoms
		peptide_interface_atoms = atom_df[
			(atom_df['chain_id'] == self.complex.peptide_chain) &
			(atom_df['residue_number'].isin(self.complex.interface_residues_peptide))
			]

		receptor_interface_atoms = atom_df[
			(atom_df['chain_id'].isin(self.complex.receptor_chains)) &
			(atom_df['residue_number'].isin(self.complex.interface_residues_receptor))
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
		if not self.complex.interface_calculated:
			return {'status': 'not_calculated'}

		peptide_atoms, receptor_atoms = self.get_interface_atoms()

		return {
			'status': 'calculated',
			'peptide_interface_residues': len(self.complex.interface_residues_peptide),
			'receptor_interface_residues': len(self.complex.interface_residues_receptor),
			'peptide_interface_atoms': len(peptide_atoms),
			'receptor_interface_atoms': len(receptor_atoms),
			'peptide_residue_list': self.complex.interface_residues_peptide,
			'receptor_residue_list': self.complex.interface_residues_receptor
		}
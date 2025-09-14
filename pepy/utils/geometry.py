"""
Geometric utilities for distance calculations and spatial operations.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from biopandas.pdb import PandasPdb


class GeometryUtils:
	"""
	Utilities for geometric calculations in protein structures.
	"""

	@staticmethod
	def get_cb_or_ca_atoms(atom_df: pd.DataFrame) -> pd.DataFrame:
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

	@staticmethod
	def calculate_interface_residues(atoms1_df: pd.DataFrame,
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
				df[col] = pd.to_numeric(df[col], errors='coerce')

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


class InterfaceFilter:
	"""
	Utilities for filtering interface results.
	"""

	@staticmethod
	def filter_by_confidence(interface_df: pd.DataFrame,
							 threshold: float = 50.0) -> pd.DataFrame:
		"""
		Filter interface residues by confidence (B-factor/pLDDT).

		Args:
			interface_df: DataFrame with interface atoms
			threshold: Minimum confidence threshold

		Returns:
			Filtered DataFrame
		"""
		return interface_df[interface_df['b_factor'] > threshold]

	@staticmethod
	def filter_by_minimum_size(interface_residues: List[int],
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
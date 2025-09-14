#!/usr/bin/env python3
"""
Tests for interface calculation functionality.
"""

import pytest
import tempfile
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from pepy import PeptideProteinComplex


class TestInterfaceCalculation:

	@pytest.fixture
	def interface_complex(self, pdb_file='pepy/tests/data/1ycr_af2_55d19_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb'):
		"""Create a complex ready for interface analysis."""
		complex = PeptideProteinComplex()
		complex.load_structure(pdb_file)
		complex.identify_chains(peptide_chain='B', receptor_chains=['A'])
		return complex

	def test_interface_calculation_basic(self, interface_complex):
		"""Test basic interface calculation."""
		pep_interface, rec_interface = interface_complex.calculate_interface()

		# Results should be stored in the complex
		assert interface_complex.interface_calculated is True
		assert interface_complex.interface_residues_peptide is not None
		assert interface_complex.interface_residues_receptor is not None

		# Should return the same results
		assert pep_interface == interface_complex.interface_residues_peptide
		assert rec_interface == interface_complex.interface_residues_receptor

	def test_interface_calculation_parameters(self, interface_complex):
		"""Test interface calculation with different parameters."""
		# Test with strict cutoffs
		pep_interface, rec_interface = interface_complex.calculate_interface(
			cb_cutoff=5.0,
			all_atom_cutoff=2.0
		)

		# Should have fewer or no interface residues with strict cutoffs
		assert isinstance(pep_interface, list)
		assert isinstance(rec_interface, list)

		# Test with relaxed cutoffs
		pep_interface2, rec_interface2 = interface_complex.calculate_interface(
			cb_cutoff=10.0,
			all_atom_cutoff=6.0
		)

		# Should have same or more interface residues with relaxed cutoffs
		assert len(pep_interface2) >= len(pep_interface)
		assert len(rec_interface2) >= len(rec_interface)

	def test_confidence_filtering(self, interface_complex):
		"""Test confidence-based filtering."""
		# Calculate without confidence filtering
		pep_interface1, rec_interface1 = interface_complex.calculate_interface(
			drop_low_confidence=False
		)

		# Calculate with confidence filtering (high threshold)
		pep_interface2, rec_interface2 = interface_complex.calculate_interface(
			drop_low_confidence=True,
			confidence_threshold=75.0
		)

		# Confidence filtering should reduce or maintain residue count
		assert len(pep_interface2) <= len(pep_interface1)
		assert isinstance(pep_interface2, list)
		assert isinstance(rec_interface2, list)

	def test_minimum_interface_size(self, interface_complex):
		"""Test minimum interface size filtering."""
		# Test with minimum size requirement
		pep_interface, rec_interface = interface_complex.calculate_interface(
			min_interface_size=10  # Very high requirement
		)

		# Should return empty if interface too small
		assert isinstance(pep_interface, list)
		assert isinstance(rec_interface, list)

	# With high minimum, likely to get empty interface

	def test_no_cb_prefiltering(self, interface_complex):
		"""Test calculation without CB prefiltering."""
		pep_interface, rec_interface = interface_complex.calculate_interface(
			cb_cutoff=-1,  # Disable CB prefiltering
			all_atom_cutoff=4.0
		)

		assert isinstance(pep_interface, list)
		assert isinstance(rec_interface, list)

	def test_no_all_atom_refinement(self, interface_complex):
		"""Test calculation without all-atom refinement."""
		pep_interface, rec_interface = interface_complex.calculate_interface(
			cb_cutoff=8.0,
			all_atom_cutoff=-1  # Disable all-atom refinement
		)

		assert isinstance(pep_interface, list)
		assert isinstance(rec_interface, list)

	def test_interface_not_ready_error(self):
		"""Test error when complex not ready for analysis."""
		complex = PeptideProteinComplex()

		with pytest.raises(RuntimeError, match="not ready for analysis"):
			complex.calculate_interface()

	def test_clear_interface_results(self, interface_complex):
		"""Test clearing interface results."""
		# Calculate interface
		interface_complex.calculate_interface()
		assert interface_complex.interface_calculated is True

		# Clear results
		interface_complex.clear_interface_results()
		assert interface_complex.interface_calculated is False
		assert interface_complex.interface_residues_peptide is None
		assert interface_complex.interface_residues_receptor is None

	def test_interface_summary(self, interface_complex):
		"""Test interface summary generation."""
		# Before calculation
		summary = interface_complex.get_interface_summary()
		assert summary['status'] == 'not_calculated'

		# After calculation
		interface_complex.calculate_interface()
		summary = interface_complex.get_interface_summary()

		assert summary['status'] == 'calculated'
		assert 'peptide_interface_residues' in summary
		assert 'receptor_interface_residues' in summary
		assert 'peptide_interface_atoms' in summary
		assert 'receptor_interface_atoms' in summary
		assert 'peptide_residue_list' in summary
		assert 'receptor_residue_list' in summary

	def test_get_interface_atoms(self, interface_complex):
		"""Test getting interface atoms."""
		# Calculate interface first
		interface_complex.calculate_interface()

		# Get all interface atoms
		pep_atoms, rec_atoms = interface_complex.get_interface_atoms('all')
		assert isinstance(pep_atoms, pd.DataFrame)
		assert isinstance(rec_atoms, pd.DataFrame)

		# Get CA atoms only
		pep_ca, rec_ca = interface_complex.get_interface_atoms('ca')
		assert isinstance(pep_ca, pd.DataFrame)
		assert isinstance(rec_ca, pd.DataFrame)

		# CA atoms should be subset of all atoms
		assert len(pep_ca) <= len(pep_atoms)
		assert len(rec_ca) <= len(rec_atoms)

		# Test other atom types
		pep_bb, rec_bb = interface_complex.get_interface_atoms('backbone')
		pep_sc, rec_sc = interface_complex.get_interface_atoms('sidechain')

		assert isinstance(pep_bb, pd.DataFrame)
		assert isinstance(rec_bb, pd.DataFrame)
		assert isinstance(pep_sc, pd.DataFrame)
		assert isinstance(rec_sc, pd.DataFrame)

	def test_interface_atoms_not_calculated_error(self, interface_complex):
		"""Test error when trying to get atoms before calculation."""
		with pytest.raises(RuntimeError, match="Interface not calculated"):
			interface_complex.get_interface_atoms()


class TestGeometryUtils:
	"""Test geometry utility functions."""

	@pytest.fixture
	def sample_atoms_df(self):
		"""Sample atom DataFrame for testing."""
		return pd.DataFrame({
			'atom_name': ['CA', 'CB', 'CA', 'CB', 'CA', 'CB'],
			'residue_name': ['ALA', 'ALA', 'GLY', 'GLY', 'VAL', 'VAL'],
			'residue_number': [1, 1, 2, 2, 3, 3],
			'chain_id': ['A', 'A', 'A', 'A', 'A', 'A'],
			'x_coord': [0.0, 1.0, 2.0, 3.0, 10.0, 15.0],
			'y_coord': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			'z_coord': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			'element_symbol': ['C', 'C', 'C', 'C', 'C', 'C'],
			'b_factor': [50.0, 50.0, 60.0, 60.0, 70.0, 80.0]
		})

	def test_get_cb_or_ca_atoms(self, sample_atoms_df):
		"""Test CB/CA atom selection."""
		from pepy.utils.geometry import GeometryUtils

		result = GeometryUtils.get_cb_or_ca_atoms(sample_atoms_df)

		# Should get CB for ALA and VAL, CA for GLY
		expected_atoms = ['CB', 'CA', 'CB']  # ALA-CB, GLY-CA, VAL-CB
		assert list(result['atom_name']) == expected_atoms

	def test_calculate_interface_residues_basic(self, sample_atoms_df):
		"""Test basic interface residue calculation."""
		from pepy.utils.geometry import GeometryUtils

		# Create two groups of atoms
		group1 = sample_atoms_df.iloc[:2].copy()  # First two atoms
		group2 = sample_atoms_df.iloc[2:4].copy()  # Next two atoms

		residues1, residues2 = GeometryUtils.calculate_interface_residues(
			group1, group2, cutoff=5.0
		)

		assert isinstance(residues1, list)
		assert isinstance(residues2, list)

	def test_interface_filter_by_confidence(self):
		"""Test confidence filtering."""
		from pepy.utils.geometry import InterfaceFilter

		# Create sample DataFrame
		df = pd.DataFrame({
			'residue_number': [1, 2, 3],
			'b_factor': [40.0, 60.0, 80.0],
			'atom_name': ['CA', 'CA', 'CA']
		})

		filtered = InterfaceFilter.filter_by_confidence(df, threshold=50.0)

		# Should keep only residues with b_factor > 50
		assert len(filtered) == 2
		assert all(filtered['b_factor'] > 50.0)

	def test_filter_by_minimum_size(self):
		"""Test minimum interface size filtering."""
		from pepy.utils.geometry import InterfaceFilter

		# Test with sufficient size
		residues = [1, 2, 3, 4, 5]
		result = InterfaceFilter.filter_by_minimum_size(residues, min_size=3)
		assert result == residues

		# Test with insufficient size
		result = InterfaceFilter.filter_by_minimum_size(residues, min_size=10)
		assert result == []


class TestInterfaceIntegration:
	"""Integration tests for complete interface workflow."""

	def test_complete_workflow(self, pdb_file='pepy/tests/data/1YCR.pdb'):
		"""Test complete workflow from loading to interface analysis."""

		# Complete workflow
		complex = PeptideProteinComplex.from_file(pdb_file)
		complex.identify_chains(peptide_chain='B')

		# Calculate interface
		pep_interface, rec_interface = complex.calculate_interface()

		# Get summary
		summary = complex.get_interface_summary()

		# Get atoms
		pep_atoms, rec_atoms = complex.get_interface_atoms()

		# Verify everything works
		assert complex.interface_calculated
		assert summary['status'] == 'calculated'
		assert isinstance(pep_atoms, pd.DataFrame)
		assert isinstance(rec_atoms, pd.DataFrame)


	def test_cache_invalidation(self, temp_file):
		"""Test that interface results are cleared when chains change."""
		pdb_content = """HEADER    TEST CACHE
ATOM      1  CA  ALA A   1      10.000   0.000   0.000  1.00 70.00           C  
ATOM      2  CA  PHE B   1      14.000   0.000   0.000  1.00 80.00           C  
ATOM      3  CA  TRP C   1      18.000   0.000   0.000  1.00 90.00           C  
END
"""

		with open(temp_file, 'w') as f:
			f.write(pdb_content)

		complex = PeptideProteinComplex.from_file(temp_file)
		complex.identify_chains(peptide_chain='B')
		complex.calculate_interface()

		print(complex.interface_calculated)
		assert complex.interface_calculated

		# Change chain assignments - should clear interface results
		complex.identify_chains(peptide_chain='C')

		# Interface results should be cleared
		print(complex.interface_calculated)
		assert not complex.interface_calculated
		assert complex.interface_residues_peptide is None
		assert complex.interface_residues_receptor is None
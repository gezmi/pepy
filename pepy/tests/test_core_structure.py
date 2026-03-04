"""
Tests for the main ProteinComplex class.
"""

import pytest
import tempfile
import os
from pepy import ProteinComplex

class TestProteinComplex:

	def test_initialization(self):
		"""Test basic initialization."""
		complex = ProteinComplex()
		assert complex.binder_chains == []
		assert complex.receptor_chains == []

	def test_load_structure_file_not_found(self):
		"""Test loading non-existent file raises error."""
		complex = ProteinComplex()
		with pytest.raises(FileNotFoundError):
			complex.load_structure("nonexistent.pdb")

	def test_load_structure_unsupported_format(self):
		"""Test loading unsupported file format."""
		complex = ProteinComplex()
		with pytest.raises(ValueError, match="Unsupported file format"):
			complex.load_structure("test.xyz")

	def test_load_pdb_from_string(self, temp_file, minimal_pdb_content):
		"""Test loading PDB from string content."""
		with open(temp_file, 'w') as f:
			f.write(minimal_pdb_content)

		complex = ProteinComplex()
		complex.load_structure(temp_file)

		assert 'ATOM' in complex.df
		assert not complex.df['ATOM'].empty
		assert 'ca_index' in complex.df['ATOM'].columns

	def test_identify_chains_auto_detect(self, temp_file, minimal_pdb_content):
		"""Test auto-detection of chains (shortest = binder)."""
		with open(temp_file, 'w') as f:
			f.write(minimal_pdb_content)

		complex = ProteinComplex()
		complex.load_structure(temp_file)
		complex.identify_chains()

		# Chain A has 2 residues, Chain B has 1 residue
		# So B should be binder (shortest)
		assert complex.binder_chains == ['B']
		assert complex.receptor_chains == ['A']

	def test_identify_chains_explicit(self, temp_file, minimal_pdb_content):
		"""Test explicit chain specification."""
		with open(temp_file, 'w') as f:
			f.write(minimal_pdb_content)

		complex = ProteinComplex()
		complex.load_structure(temp_file)
		complex.identify_chains(binder_chains='A', receptor_chains=['B'])

		assert complex.binder_chains == ['A']
		assert complex.receptor_chains == ['B']

	def test_identify_chains_multi_binder(self, temp_file, minimal_pdb_content):
		"""Test specifying multi-chain binder as list."""
		with open(temp_file, 'w') as f:
			f.write(minimal_pdb_content)

		complex = ProteinComplex()
		complex.load_structure(temp_file)
		complex.identify_chains(binder_chains=['A'], receptor_chains=['B'])

		assert complex.binder_chains == ['A']
		assert complex.receptor_chains == ['B']

	def test_identify_chains_invalid_binder(self, temp_file, minimal_pdb_content):
		"""Test error when specifying invalid binder chain."""
		with open(temp_file, 'w') as f:
			f.write(minimal_pdb_content)

		complex = ProteinComplex()
		complex.load_structure(temp_file)

		with pytest.raises(ValueError, match="Binder chain 'C' not found"):
			complex.identify_chains(binder_chains='C')

	def test_get_chain_lengths(self, temp_file, minimal_pdb_content):
		"""Test chain length calculation."""
		with open(temp_file, 'w') as f:
			f.write(minimal_pdb_content)

		complex = ProteinComplex()
		complex.load_structure(temp_file)

		lengths = complex.get_chain_lengths()
		assert lengths['A'] == 2  # ALA, GLY
		assert lengths['B'] == 1  # VAL

	def test_get_chain_sequence(self, temp_file, minimal_pdb_content):
		"""Test sequence extraction."""
		with open(temp_file, 'w') as f:
			f.write(minimal_pdb_content)

		complex = ProteinComplex()
		complex.load_structure(temp_file)

		chain_info = complex.get_chain_info()
		assert chain_info['A']['sequence'] == 'AG'  # ALA-GLY
		assert chain_info['B']['sequence'] == 'V'  # VAL

	def test_analysis_summary(self, temp_file, minimal_pdb_content):
		"""Test analysis summary generation."""
		with open(temp_file, 'w') as f:
			f.write(minimal_pdb_content)

		complex = ProteinComplex()
		complex.load_structure(temp_file)

		# Before chain identification
		summary = complex.get_analysis_summary()
		assert summary['status'] == 'not_ready'

		# After chain identification
		complex.identify_chains()
		summary = complex.get_analysis_summary()
		assert summary['status'] == 'ready'
		assert summary['binder_chains'] == ['B']
		assert summary['binder_length'] == 1

	def test_cache_clearing(self, temp_file, minimal_pdb_content):
		"""Test that cache gets cleared appropriately."""
		with open(temp_file, 'w') as f:
			f.write(minimal_pdb_content)

		complex = ProteinComplex()
		complex.load_structure(temp_file)

		# Populate cache
		lengths = complex.get_chain_lengths()
		assert complex._cache['chain_lengths']  # Cache populated

		# Clear cache
		complex.clear_cache()
		assert not complex._cache['chain_lengths']  # Cache cleared

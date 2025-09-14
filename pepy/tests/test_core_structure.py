"""
Tests for the main PeptideProteinComplex class.
"""

import pytest
import tempfile
import os
from pepy import PeptideProteinComplex

class TestPeptideProteinComplex:

	def test_initialization(self):
		"""Test basic initialization."""
		complex = PeptideProteinComplex()
		assert complex.peptide_chain is None
		assert complex.receptor_chains == []
		assert complex.cb_cutoff == 8.0
		assert complex.all_atom_cutoff == 4.0

	def test_load_structure_file_not_found(self):
		"""Test loading non-existent file raises error."""
		complex = PeptideProteinComplex()
		with pytest.raises(FileNotFoundError):
			complex.load_structure("nonexistent.pdb")

	def test_load_structure_unsupported_format(self):
		"""Test loading unsupported file format."""
		complex = PeptideProteinComplex()
		with pytest.raises(ValueError, match="Unsupported file format"):
			complex.load_structure("test.xyz")

	def test_load_pdb_from_string(self, temp_file, minimal_pdb_content):
		"""Test loading PDB from string content."""
		# Write content to temp file
		with open(temp_file, 'w') as f:
			f.write(minimal_pdb_content)

		# Load structure
		complex = PeptideProteinComplex()
		complex.load_structure(temp_file)

		# Check structure loaded
		assert 'ATOM' in complex.df
		assert not complex.df['ATOM'].empty
		assert 'ca_index' in complex.df['ATOM'].columns

	def test_identify_chains_auto_detect(self, temp_file, minimal_pdb_content):
		"""Test auto-detection of chains (shortest = peptide)."""
		with open(temp_file, 'w') as f:
			f.write(minimal_pdb_content)

		complex = PeptideProteinComplex()
		complex.load_structure(temp_file)
		complex.identify_chains()

		# Chain A has 2 residues, Chain B has 1 residue
		# So B should be peptide (shortest)
		assert complex.peptide_chain == 'B'
		assert complex.receptor_chains == ['A']

	def test_identify_chains_explicit(self, temp_file, minimal_pdb_content):
		"""Test explicit chain specification."""
		with open(temp_file, 'w') as f:
			f.write(minimal_pdb_content)

		complex = PeptideProteinComplex()
		complex.load_structure(temp_file)
		complex.identify_chains(peptide_chain='A', receptor_chains=['B'])

		assert complex.peptide_chain == 'A'
		assert complex.receptor_chains == ['B']

	def test_identify_chains_invalid_peptide(self, temp_file, minimal_pdb_content):
		"""Test error when specifying invalid peptide chain."""
		with open(temp_file, 'w') as f:
			f.write(minimal_pdb_content)

		complex = PeptideProteinComplex()
		complex.load_structure(temp_file)

		with pytest.raises(ValueError, match="Peptide chain 'C' not found"):
			complex.identify_chains(peptide_chain='C')

	def test_get_chain_lengths(self, temp_file, minimal_pdb_content):
		"""Test chain length calculation."""
		with open(temp_file, 'w') as f:
			f.write(minimal_pdb_content)

		complex = PeptideProteinComplex()
		complex.load_structure(temp_file)

		lengths = complex.get_chain_lengths()
		assert lengths['A'] == 2  # ALA, GLY
		assert lengths['B'] == 1  # VAL

	def test_get_chain_sequence(self, temp_file, minimal_pdb_content):
		"""Test sequence extraction."""
		with open(temp_file, 'w') as f:
			f.write(minimal_pdb_content)

		complex = PeptideProteinComplex()
		complex.load_structure(temp_file)

		chain_info = complex.get_chain_info()
		assert chain_info['A']['sequence'] == 'AG'  # ALA-GLY
		assert chain_info['B']['sequence'] == 'V'  # VAL

	def test_analysis_summary(self, temp_file, minimal_pdb_content):
		"""Test analysis summary generation."""
		with open(temp_file, 'w') as f:
			f.write(minimal_pdb_content)

		complex = PeptideProteinComplex()
		complex.load_structure(temp_file)

		# Before chain identification
		summary = complex.get_analysis_summary()
		assert summary['status'] == 'not_ready'

		# After chain identification
		complex.identify_chains()
		summary = complex.get_analysis_summary()
		assert summary['status'] == 'ready'
		assert summary['peptide_chain'] == 'B'
		assert summary['peptide_length'] == 1

	def test_cache_clearing(self, temp_file, minimal_pdb_content):
		"""Test that cache gets cleared appropriately."""
		with open(temp_file, 'w') as f:
			f.write(minimal_pdb_content)

		complex = PeptideProteinComplex()
		complex.load_structure(temp_file)

		# Populate cache
		lengths = complex.get_chain_lengths()
		assert complex._cache['chain_lengths']  # Cache populated

		# Clear cache
		complex.clear_cache()
		assert not complex._cache['chain_lengths']  # Cache cleared
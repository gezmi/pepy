"""
Structure file loading utilities.
"""

import os
import tempfile
import shutil
import pandas as pd
from biopandas.pdb import PandasPdb
from biopandas.mmcif import PandasMmcif
from typing import Dict, Any

from ..core.base import AF3_MISSING_COLUMNS


def load_pdb_file(filepath: str) -> Dict[str, pd.DataFrame]:
	"""
	Load PDB file and return dataframes.

	Args:
		filepath: Path to PDB file

	Returns:
		Dictionary of dataframes from biopandas
	"""
	pdb_df = PandasPdb().read_pdb(filepath)
	return pdb_df.df


def load_cif_file(filepath: str) -> Dict[str, pd.DataFrame]:
	"""
	Load CIF file and convert to PDB format dataframes.

	Args:
		filepath: Path to CIF file

	Returns:
		Dictionary of dataframes in PDB format
	"""
	mmcif_df = PandasMmcif().read_mmcif(filepath)

	# Fix column mappings for AF3 compatibility
	for df_type in ['ATOM', 'HETATM']:
		if df_type in mmcif_df.df and not mmcif_df.df[df_type].empty:
			mmcif_df.df[df_type]['auth_atom_id'] = mmcif_df.df[df_type]['label_atom_id']
			mmcif_df.df[df_type]['auth_comp_id'] = mmcif_df.df[df_type]['label_comp_id']

	# Convert to PandasPdb format
	pdb_df = mmcif_df.convert_to_pandas_pdb(offset_chains=False)

	# Fix occupancy column type
	for df_type in ['ATOM', 'HETATM']:
		if df_type in pdb_df.df and not pdb_df.df[df_type].empty:
			pdb_df.df[df_type]['occupancy'] = pd.to_numeric(
				pdb_df.df[df_type]['occupancy'], errors='coerce'
			)

	return pdb_df.df


def fix_af3_file_safe(cif_file: str) -> None:
	"""
	Safely fix AF3 CIF files by adding missing columns.
	Uses temporary file approach to avoid corruption.

	Args:
		cif_file: Path to CIF file to fix

	Raises:
		RuntimeError: If file processing fails
	"""
	# Create temporary file
	with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.cif') as temp_file:
		temp_path = temp_file.name

	try:
		# Read original file
		with open(cif_file, 'r') as f:
			lines = f.read()

		# Process lines
		new_lines = []
		inside_loop = False
		headers = []

		for line in lines.split('\n'):
			if line.startswith('#'):
				inside_loop = False
				new_lines.append(line)
				continue

			if line.startswith('_atom_site.'):
				headers.append(line.rstrip())
				continue

			if (line.startswith('ATOM') or line.startswith('HETATM')) and headers and not inside_loop:
				# Add missing columns to headers
				for col in AF3_MISSING_COLUMNS:
					if col not in headers:
						headers.append(col)
				new_lines.extend(headers)
				inside_loop = True

				# Get indices for filling missing data
				comp_idx = headers.index('_atom_site.label_comp_id')
				atom_idx = headers.index('_atom_site.label_atom_id')

			if inside_loop and (line.startswith('ATOM') or line.startswith('HETATM')):
				columns = line.split()
				# Add values for missing columns
				while len(columns) < len(headers):
					missing_col = headers[len(columns)]
					if missing_col == '_atom_site.auth_comp_id':
						columns.append(columns[comp_idx])
					elif missing_col == '_atom_site.auth_atom_id':
						columns.append(columns[atom_idx])
					else:
						columns.append('?')
				new_lines.append(' '.join(columns))
			else:
				new_lines.append(line)

		# Write to temporary file
		with open(temp_path, 'w') as f:
			f.write('\n'.join(new_lines))

		# Replace original file atomically
		shutil.move(temp_path, cif_file)

	except Exception as e:
		# Clean up temp file if it exists
		if os.path.exists(temp_path):
			os.unlink(temp_path)
		raise RuntimeError(f"Failed to fix AF3 file {cif_file}: {e}")


def add_ca_indices(atom_df: pd.DataFrame) -> pd.DataFrame:
	"""
	Add CA indices for PAE matrix mapping.

	Args:
		atom_df: ATOM dataframe

	Returns:
		Modified dataframe with ca_index column
	"""
	atom_df = atom_df.copy()
	atom_df['ca_index'] = (atom_df['atom_name'] == 'CA').cumsum() - 1
	return atom_df
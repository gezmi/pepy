"""
Base constants and utilities for peptide-protein analysis.
"""

# AF3 CIF file missing columns that need to be added
AF3_MISSING_COLUMNS = [
    '_atom_site.auth_atom_id',
    '_atom_site.auth_comp_id',
    '_atom_site.pdbx_formal_charge'
]

# Default analysis parameters
DEFAULT_CB_CUTOFF = 8.0
DEFAULT_ALL_ATOM_CUTOFF = 4.0

# Cache keys for consistent naming
CACHE_KEYS = {
    'chain_lengths': 'chain_lengths',
    'ca_indices': 'ca_indices',
    'interface_cb': 'interface_cb',
    'interface_all_atom': 'interface_all_atom',
}
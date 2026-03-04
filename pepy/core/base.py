"""
Base constants and utilities for protein complex analysis.
"""

# AF3 CIF file missing columns that need to be added
AF3_MISSING_COLUMNS = [
    '_atom_site.auth_atom_id',
    '_atom_site.auth_comp_id',
    '_atom_site.pdbx_formal_charge'
]

# Cache keys for consistent naming
CACHE_KEYS = {
    'chain_lengths': 'chain_lengths',
    'ca_indices': 'ca_indices',
    'interface_cb': 'interface_cb',
    'interface_all_atom': 'interface_all_atom',
}
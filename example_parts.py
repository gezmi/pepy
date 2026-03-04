"""
Step-by-step example — each part used independently.

Shows that interface, confidence, and metrics are fully separable.
"""
from pepy import ProteinComplex

# =============================================================================
# 1. LOAD STRUCTURE
# =============================================================================
complex = ProteinComplex.from_file(
    "pepy/tests/data/1ycr_af2_55d19_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb"
)

# Inspect what's in the file
print("Available chains:", complex.get_available_chains())
print("Chain lengths:", complex.get_chain_lengths())

# =============================================================================
# 2. ASSIGN CHAINS
# =============================================================================
# Auto-detect: shortest chain → binder
complex.identify_chains()

# Or explicit (string or list — both work):
# complex.identify_chains(binder_chains='B', receptor_chains=['A'])
# complex.identify_chains(binder_chains=['B', 'C'])      # multi-chain binder

print(f"Binder:   {complex.binder_chains}")
print(f"Receptor: {complex.receptor_chains}")

# Chain info (lengths, sequences, roles)
for chain_id, info in complex.get_chain_info().items():
    print(f"  {chain_id}: {info['length']} residues, role={info['type']}, seq={info['sequence'][:20]}...")

# =============================================================================
# 3. INTERFACE CALCULATION (works without any confidence data)
# =============================================================================
binder_res, receptor_res = complex.calculate_interface(
    cb_cutoff=8.0,              # CB prefilter distance (Å), -1 to skip
    all_atom_cutoff=4.0,        # all-atom refinement distance (Å), -1 to skip
    min_interface_size=1,       # min binder residues to count as interface
    drop_low_confidence=False,  # filter by pLDDT (b-factor)
)

print(f"\nInterface residues:")
print(f"  Binder:   {binder_res}")
print(f"  Receptor: {receptor_res}")

# Summary dict
summary = complex.get_interface_summary()
print(f"  Binder interface size:   {summary['binder_interface_residues']}")
print(f"  Receptor interface size: {summary['receptor_interface_residues']}")

# =============================================================================
# 4. ACCESS INTERFACE ATOMS (DataFrames for downstream analysis)
# =============================================================================
binder_atoms, receptor_atoms = complex.get_interface_atoms('all')
print(f"\nInterface atoms: {len(binder_atoms)} binder, {len(receptor_atoms)} receptor")

# Filter by atom type
binder_ca, receptor_ca = complex.get_interface_atoms('ca')
binder_bb, receptor_bb = complex.get_interface_atoms('backbone')
binder_sc, receptor_sc = complex.get_interface_atoms('sidechain')

print(f"  CA only:      {len(binder_ca)} + {len(receptor_ca)}")
print(f"  Backbone:     {len(binder_bb)} + {len(receptor_bb)}")
print(f"  Sidechain:    {len(binder_sc)} + {len(receptor_sc)}")

# These are standard pandas DataFrames — use for your own analysis
# e.g. binder_ca[['chain_id', 'residue_number', 'x_coord', 'y_coord', 'z_coord']]

# =============================================================================
# 5. CONFIDENCE DATA (optional — loads from JSON/NPZ next to structure file)
# =============================================================================
loaded = complex.load_confidence_data()
if loaded:
    conf = complex.get_confidence_summary()
    print(f"\nConfidence data:")
    print(f"  iPTM: {conf.get('iptm', 'N/A')}")
    print(f"  pTM:  {conf.get('ptm', 'N/A')}")
    print(f"  PAE matrix available: {conf['has_pae_matrix']}")

    # Raw data access
    # complex.confidence_data['pae_matrix']   — pandas DataFrame
    # complex.confidence_data['iptm']          — float
    # complex.confidence_data['ptm']           — float
    # complex.confidence_data['confidence']    — float (0.8*iptm + 0.2*ptm)
else:
    print("\nNo confidence file found (OK — interface still works)")

# =============================================================================
# 6. CONFIDENCE METRICS (requires both interface + confidence data)
# =============================================================================
if complex.has_confidence_data and complex.interface_calculated:
    metrics = complex.calculate_confidence_metrics()
    print(f"\nInterface confidence metrics:")
    print(f"  Avg pLDDT (interface): {metrics['avg_plddt_interface']}")
    print(f"  Max pLDDT (interface): {metrics['max_plddt_interface']}")
    print(f"  Interface PAE:         {metrics['interface_pae']}")
    print(f"  iPTM:                  {metrics['iptm']}")

"""
Format-specific examples — AF2/ColabFold, AF3, ChAI, PDB fetch.

Shows that the same API works regardless of source format.
Confidence file finding is automatic based on naming conventions.
"""
from pepy import ProteinComplex

# =============================================================================
# AlphaFold2 / ColabFold  (.pdb + _scores.json or similar)
# =============================================================================
print("=== AF2 / ColabFold ===")
af2 = ProteinComplex.from_file(
    "pepy/tests/data/1ycr_af2_55d19_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb"
)
af2.identify_chains()
af2.calculate_interface()
af2.load_confidence_data()  # looks for matching JSON automatically
print(f"  Interface: {len(af2.interface_residues_binder)} binder residues")
print(f"  Confidence loaded: {af2.has_confidence_data}")

# =============================================================================
# AlphaFold3  (.cif + _full_data_0.json / _summary_confidences_0.json)
# =============================================================================
print("\n=== AF3 ===")
af3 = ProteinComplex.from_file(
    "pepy/tests/data/fold_1ycr_af3_model_0.cif"
)
af3.identify_chains()
af3.calculate_interface()
af3.load_confidence_data()  # finds AF3 JSON pair automatically
print(f"  Interface: {len(af3.interface_residues_binder)} binder residues")
print(f"  Confidence loaded: {af3.has_confidence_data}")

# =============================================================================
# ChAI  (.pdb + .npz)
# =============================================================================
print("\n=== ChAI ===")
chai = ProteinComplex.from_file(
    "pepy/tests/data/pred.model_idx_0.pdb"
)
chai.identify_chains()
chai.calculate_interface()
chai.load_confidence_data()  # finds matching .npz automatically
print(f"  Interface: {len(chai.interface_residues_binder)} binder residues")
print(f"  Confidence loaded: {chai.has_confidence_data}")

# =============================================================================
# Fetch from PDB  (no confidence data expected)
# =============================================================================
print("\n=== PDB fetch ===")
pdb = ProteinComplex()
pdb.fetch_pdb('1ycr')
pdb.identify_chains(binder_chains='B')
pdb.calculate_interface()
print(f"  Interface: {len(pdb.interface_residues_binder)} binder residues")

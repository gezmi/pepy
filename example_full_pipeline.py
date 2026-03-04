"""
Full pipeline example — runs everything in one call.

calculate_all_metrics() does: interface → confidence loading → metrics
"""
from pepy import ProteinComplex

# --- Load & identify chains -------------------------------------------------
complex = ProteinComplex.from_file(
    "pepy/tests/data/1ycr_af2_55d19_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb"
)
complex.identify_chains()                          # auto-detect (shortest = binder)
# complex.identify_chains(binder_chains='B')       # or set explicitly

# --- One call does it all ----------------------------------------------------
results = complex.calculate_all_metrics(
    cb_cutoff=8.0,
    all_atom_cutoff=4.0,
    confidence_threshold=70.0,
    require_confidence=False,       # don't fail if no JSON found
)

# --- Print results -----------------------------------------------------------
metrics = results['metrics']
if metrics['status'] == 'calculated':
    print(f"Interface pLDDT:  {metrics['avg_plddt_interface']}")
    print(f"Interface PAE:    {metrics['interface_pae']}")
    print(f"iPTM:             {metrics['iptm']}")
    print(f"Confidence:       {metrics['combined_confidence']}")
    n_binder = len(results['interface_residues']['binder'])
    n_receptor = len(results['interface_residues']['receptor'])
    print(f"Interface size:   {n_binder} binder + {n_receptor} receptor residues")
else:
    print(f"Status: {metrics['status']}")

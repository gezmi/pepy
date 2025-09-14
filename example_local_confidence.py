from pepy import PeptideProteinComplex

# Load structure and identify chains
complex = PeptideProteinComplex.from_file("pepy/tests/data/1ycr_af2_55d19_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb")
complex.identify_chains()

print(f"Chains: Peptide={complex.peptide_chain}, Receptor={complex.receptor_chains}")

# Complete analysis workflow
results = complex.calculate_all_metrics(
    cb_cutoff=8.0,
    all_atom_cutoff=4.0,
    confidence_threshold=70.0,
    require_confidence=False
)

# Display key results
metrics = results['metrics']
if metrics['status'] == 'calculated':
    print(f"Interface pLDDT: {metrics['avg_plddt_interface']}")
    print(f"Interface PAE: {metrics['interface_pae']}")
    print(f"iPTM: {metrics['iptm']}")
    print(f"Combined confidence: {metrics['combined_confidence']}")
    print(f"Interface size: {len(results['interface_residues']['peptide'])} peptide, {len(results['interface_residues']['receptor'])} receptor residues")
else:
    print(f"Analysis status: {metrics['status']}")
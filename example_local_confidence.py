from pepy import PeptideProteinComplex

# Load and prepare structure
complex = PeptideProteinComplex.from_file("pepy/tests/data/1ycr_af2_55d19_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb")
complex.identify_chains()  # Auto-detect or specify chains

# Calculate interface with default parameters
pep_interface, rec_interface = complex.calculate_interface()

# Or with custom parameters
pep_interface, rec_interface = complex.calculate_interface(
    cb_cutoff=8.0,
    all_atom_cutoff=4.0,
    min_interface_size=3,
    drop_low_confidence=True,
    confidence_threshold=60.0
)

# Access results
print(f"Peptide interface: {complex.interface_residues_peptide}")
print(f"Receptor interface: {complex.interface_residues_receptor}")

# Get summary
summary = complex.get_interface_summary()
print(summary)

success = complex.load_confidence_data()
if success:
    summary_confidence = complex.get_confidence_summary()
    print(f"Confidence data: {summary_confidence}")

    # Calculate interface with confidence filtering available
    pep_interface, rec_interface = complex.calculate_interface(
        drop_low_confidence=True,
        confidence_threshold=70.0
    )
else:
    print("No confidence data found, continuing without confidence metrics")
    pep_interface, rec_interface = complex.calculate_interface()
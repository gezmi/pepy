from pepy import PeptideProteinComplex

# Create complex and load from PDB ID using built-in BioPandas functionality
complex = PeptideProteinComplex()
complex.fetch_pdb('1ycr')  # Downloads and loads PDB structure

# Identify chains and calculate interface
complex.identify_chains(peptide_chain='B', receptor_chains=['A']) # if setting chains manually
# OR:
complex.identify_chains() # automatic detection
pep_interface, rec_interface = complex.calculate_interface(
    cb_cutoff=8.0,
    all_atom_cutoff=4
)

# Get summary
summary = complex.get_interface_summary()

success = complex.load_confidence_data()
if success:
    summary = complex.get_confidence_summary()
    print(f"Confidence data: {summary}")

    # Calculate interface with confidence filtering available
    pep_interface, rec_interface = complex.calculate_interface(
        drop_low_confidence=True,
        confidence_threshold=70.0
    )
else:
    print("No confidence data found, continuing without confidence metrics")
    pep_interface, rec_interface = complex.calculate_interface()

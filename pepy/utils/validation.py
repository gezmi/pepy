"""
Validation utilities for peptide-protein complex analysis.
"""

import os
from typing import List, Optional


def validate_file_path(filepath: str) -> None:
    """
    Validate that file exists and has supported extension.

    Args:
        filepath: Path to structure file

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format not supported
    """
    # Check format first (so we can test unsupported formats without creating files)
    if not (filepath.endswith('.pdb') or filepath.endswith('.cif')):
        raise ValueError(f"Unsupported file format: {filepath}. Supported: .pdb, .cif")

    # Then check existence
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Structure file not found: {filepath}")


def validate_chains(available_chains: List[str],
                   peptide_chain: Optional[str] = None,
                   receptor_chains: Optional[List[str]] = None) -> None:
    """
    Validate chain assignments against available chains.

    Args:
        available_chains: List of chains present in structure
        peptide_chain: Proposed peptide chain ID
        receptor_chains: Proposed receptor chain IDs

    Raises:
        ValueError: If validation fails
    """
    if not available_chains:
        raise ValueError("No chains found in structure")

    if peptide_chain and peptide_chain not in available_chains:
        raise ValueError(f"Peptide chain '{peptide_chain}' not found in structure. "
                        f"Available chains: {available_chains}")

    if receptor_chains:
        invalid_receptors = [c for c in receptor_chains if c not in available_chains]
        if invalid_receptors:
            raise ValueError(f"Receptor chains not found: {invalid_receptors}. "
                           f"Available chains: {available_chains}")

        if peptide_chain and peptide_chain in receptor_chains:
            raise ValueError(f"Chain '{peptide_chain}' cannot be both peptide and receptor")


def validate_analysis_ready(peptide_chain: Optional[str],
                           receptor_chains: List[str],
                           atom_df_empty: bool) -> None:
    """
    Validate that structure is ready for analysis.

    Args:
        peptide_chain: Assigned peptide chain
        receptor_chains: Assigned receptor chains
        atom_df_empty: Whether ATOM dataframe is empty

    Raises:
        RuntimeError: If structure not ready for analysis
    """
    if peptide_chain is None:
        raise RuntimeError("No peptide chain assigned. Call identify_chains() first.")

    if not receptor_chains:
        raise RuntimeError("No receptor chains assigned. Call identify_chains() first.")

    if atom_df_empty:
        raise RuntimeError("No ATOM records found in structure.")
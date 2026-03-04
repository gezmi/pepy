"""
Validation utilities for protein complex analysis.
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
                   binder_chains: Optional[List[str]] = None,
                   receptor_chains: Optional[List[str]] = None) -> None:
    """
    Validate chain assignments against available chains.

    Args:
        available_chains: List of chains present in structure
        binder_chains: Proposed binder chain IDs
        receptor_chains: Proposed receptor chain IDs

    Raises:
        ValueError: If validation fails
    """
    if not available_chains:
        raise ValueError("No chains found in structure")

    if binder_chains:
        for bc in binder_chains:
            if bc not in available_chains:
                raise ValueError(f"Binder chain '{bc}' not found in structure. "
                                f"Available chains: {available_chains}")

    if receptor_chains:
        invalid_receptors = [c for c in receptor_chains if c not in available_chains]
        if invalid_receptors:
            raise ValueError(f"Receptor chains not found: {invalid_receptors}. "
                           f"Available chains: {available_chains}")

        if binder_chains:
            overlap = set(binder_chains) & set(receptor_chains)
            if overlap:
                raise ValueError(f"Chain(s) {overlap} cannot be both binder and receptor")

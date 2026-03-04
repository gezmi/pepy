"""
Pytest configuration and shared fixtures.
"""

import pytest
import tempfile
import os
from pathlib import Path

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"

@pytest.fixture
def sample_pdb_path():
    """Path to sample PDB file."""
    return TEST_DATA_DIR / "sample.pdb"

@pytest.fixture
def sample_cif_path():
    """Path to sample CIF file."""
    return TEST_DATA_DIR / "sample.cif"

@pytest.fixture
def temp_file():
    """Create temporary file that gets cleaned up."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdb') as f:
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.unlink(temp_path)

@pytest.fixture
def minimal_pdb_content():
    """Minimal PDB content for testing."""
    return """HEADER    TEST STRUCTURE
ATOM      1  N   ALA A   1      20.154  16.967  10.000  1.00 50.00           N  
ATOM      2  CA  ALA A   1      18.831  16.239  10.000  1.00 50.00           C  
ATOM      3  N   GLY A   2      17.654  17.067  10.000  1.00 60.00           N  
ATOM      4  CA  GLY A   2      16.331  16.339  10.000  1.00 60.00           C  
ATOM      5  N   VAL B   1      15.154  17.167  10.000  1.00 70.00           N  
ATOM      6  CA  VAL B   1      13.831  16.439  10.000  1.00 70.00           C  
END
"""
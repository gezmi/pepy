# PePy: Protein Complex Interface Analysis

PePy makes it easy to calculate and analyze **protein-protein interfaces** from structure files — with optional confidence metrics from AlphaFold2, AlphaFold3, and ChAI predictions.

Built on top of [BioPandas](https://github.com/gezmi/biopandas), all data lives in pandas DataFrames.

## Features

- **Interface calculation** — two-stage CB prefilter + all-atom refinement
- **Confidence metrics** — iPTM, pTM, interface pLDDT, interface PAE
- **Auto-detection** — finds confidence JSON/NPZ files automatically (AF2, AF3, ChAI)
- **Multi-chain support** — binder and receptor can each span multiple chains
- **DataFrame access** — interface atoms returned as pandas DataFrames for downstream analysis

## Installation

```bash
pip install -e .
```

Requires the [BioPandas fork](https://github.com/gezmi/biopandas/tree/stack_tmalign) (installed automatically from `pyproject.toml`).

## Quick Start

```python
from pepy import ProteinComplex

# Load structure and auto-detect chains (shortest = binder)
cx = ProteinComplex.from_file("structure.pdb")
cx.identify_chains()

# Calculate interface
binder_res, receptor_res = cx.calculate_interface()

# Load confidence data (if available next to the structure file)
cx.load_confidence_data()

# Get metrics
metrics = cx.calculate_confidence_metrics()
print(metrics['avg_plddt_interface'], metrics['interface_pae'], metrics['iptm'])
```

Or run everything in one call:

```python
results = cx.calculate_all_metrics()
```

## Tutorials

Jupyter notebook tutorials are in [`docs/tutorials/`](docs/tutorials/):

| Notebook | Description |
|----------|-------------|
| [Working with Protein Complexes](docs/tutorials/Working_with_Protein_Complexes.ipynb) | Loading structures, chain assignment, interface calculation, accessing atoms |
| [Working with Confidence Data](docs/tutorials/Working_with_Confidence_Data.ipynb) | AF2/AF3/ChAI confidence loading, iPTM/pTM, PAE matrices, interface metrics |
| [Full Analysis Pipeline](docs/tutorials/Full_Analysis_Pipeline.ipynb) | One-call workflow, batch processing, combining results |

## Command Line

After `pip install -e .`, PePy is available as a CLI tool:

```bash
# Single file
pepy -i structure.pdb -o results.tsv

# With confidence metrics
pepy -i structure.pdb -o results.tsv -j

# Batch — glob pattern
pepy -g "predictions/*.pdb" -o results.tsv -j

# Batch — file list
pepy -l file_list.txt -o results.tsv -j

# Explicit chains, custom cutoffs, drop low confidence
pepy -i structure.pdb -o results.tsv -b B -r A --cb-cutoff 8 --all-atom-cutoff 4 -d -j

# Parallel processing
pepy -g "predictions/*.pdb" -o results.tsv -j -c -1
```

Output is a tab-separated file with columns: `file`, `binder`, `receptor`, `n_binder_res`, `n_receptor_res`, `avg_plddt`, `max_plddt`, and (with `-j`): `iptm`, `ptm`, `ipae`, `min_ipae`, `confidence`.

## Supported Formats

| Source | Structure | Confidence |
|--------|-----------|------------|
| AlphaFold2 / ColabFold | `.pdb` | `_scores*.json` |
| AlphaFold3 | `.cif` | `_full_data_*.json` + `_summary_confidences_*.json` |
| ChAI | `.pdb` | `scores.*.npz` |
| PDB / custom | `.pdb`, `.cif` | — |

## API Overview

```python
from pepy import ProteinComplex

cx = ProteinComplex.from_file("structure.pdb")   # or .cif
cx.fetch_pdb("1YCR")                              # or fetch from RCSB

# Chain assignment
cx.identify_chains()                               # auto-detect
cx.identify_chains(binder_chains='B')              # explicit (string)
cx.identify_chains(binder_chains=['B', 'C'])       # multi-chain binder

# Interface
binder_res, rec_res = cx.calculate_interface(
    cb_cutoff=8.0, all_atom_cutoff=4.0,
    drop_low_confidence=True, confidence_threshold=70.0,
)
cx.get_interface_summary()
cx.get_interface_atoms('ca')       # 'all', 'backbone', 'sidechain', 'ca'

# Confidence
cx.load_confidence_data()          # auto-finds JSON/NPZ
cx.get_confidence_summary()
cx.calculate_confidence_metrics()

# All-in-one
cx.calculate_all_metrics()
```

## Requirements

- Python ≥ 3.9
- pandas, numpy, ujson
- [BioPandas fork](https://github.com/gezmi/biopandas/tree/stack_tmalign)

## Author

Julia K. Varga — jvarga92@gmail.com

"""
Command-line interface for pepy.

Usage:
    pepy -i structure.pdb -o results.tsv
    pepy -l file_list.txt -o results.tsv --binder B --receptor A
    pepy -i structure.cif -o results.tsv --confidence
"""

import argparse
import glob
import os
import sys
import pandas as pd
from typing import List, Dict, Any

from pepy import ProteinComplex


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog='pepy',
        description='Calculate protein complex interface metrics and confidence scores.',
    )

    # Input (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '-i', '--input',
        help='Input structure file (.pdb or .cif)',
    )
    input_group.add_argument(
        '-l', '--list',
        help='Text file with one structure path per line',
    )
    input_group.add_argument(
        '-g', '--glob',
        help='Glob pattern for structure files (e.g. "results/*.pdb")',
    )

    # Output
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output TSV file',
    )

    # Chain assignment
    parser.add_argument(
        '-b', '--binder',
        default=None,
        help='Binder chain(s), comma-separated (default: auto-detect shortest)',
    )
    parser.add_argument(
        '-r', '--receptor',
        default=None,
        help='Receptor chain(s), comma-separated (default: all non-binder)',
    )

    # Interface parameters
    parser.add_argument(
        '--cb-cutoff', type=float, default=8.0,
        help='CB prefilter distance cutoff in Å (default: 8.0, -1 to disable)',
    )
    parser.add_argument(
        '--all-atom-cutoff', type=float, default=4.0,
        help='All-atom refinement distance cutoff in Å (default: 4.0, -1 to disable)',
    )
    parser.add_argument(
        '--min-interface', type=int, default=1,
        help='Minimum binder residues to form interface (default: 1)',
    )
    parser.add_argument(
        '-d', '--drop-low-confidence', action='store_true',
        help='Drop binder residues with pLDDT < threshold',
    )
    parser.add_argument(
        '--confidence-threshold', type=float, default=50.0,
        help='pLDDT threshold for --drop-low-confidence (default: 50.0)',
    )

    # Confidence
    parser.add_argument(
        '-j', '--confidence', action='store_true',
        help='Load and report confidence metrics (iPTM, iPAE, etc.)',
    )
    parser.add_argument(
        '--require-confidence', action='store_true',
        help='Fail if confidence data not found (default: skip gracefully)',
    )

    # Parallelism
    parser.add_argument(
        '-c', '--cpu', type=int, default=1,
        help='Number of parallel workers (default: 1, -1 for all cores)',
    )

    return parser.parse_args(argv)


def collect_files(args) -> List[str]:
    """Collect structure file paths from input arguments."""
    if args.input:
        return [args.input]
    elif args.list:
        with open(args.list, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    elif args.glob:
        files = sorted(glob.glob(args.glob))
        if not files:
            print(f'No files matched pattern: {args.glob}', file=sys.stderr)
            sys.exit(1)
        return files


def process_one(filepath: str, args) -> Dict[str, Any]:
    """Process a single structure file and return a result row."""
    row = {'file': os.path.basename(filepath)}

    try:
        cx = ProteinComplex.from_file(filepath)

        # Chain assignment
        binder = args.binder.split(',') if args.binder else None
        receptor = args.receptor.split(',') if args.receptor else None
        cx.identify_chains(binder_chains=binder, receptor_chains=receptor)

        row['binder'] = ','.join(cx.binder_chains)
        row['receptor'] = ','.join(cx.receptor_chains)

        # Interface
        binder_res, rec_res = cx.calculate_interface(
            cb_cutoff=args.cb_cutoff,
            all_atom_cutoff=args.all_atom_cutoff,
            min_interface_size=args.min_interface,
            drop_low_confidence=args.drop_low_confidence,
            confidence_threshold=args.confidence_threshold,
        )
        row['n_binder_res'] = len(binder_res)
        row['n_receptor_res'] = len(rec_res)

        # pLDDT from B-factors (always available)
        if binder_res:
            binder_ca, _ = cx.get_interface_atoms('ca')
            row['avg_plddt'] = round(float(binder_ca['b_factor'].mean()), 2)
            row['max_plddt'] = round(float(binder_ca['b_factor'].max()), 2)
        else:
            row['avg_plddt'] = 0.0
            row['max_plddt'] = 0.0

        # Confidence metrics
        if args.confidence:
            loaded = cx.load_confidence_data(
                require_confidence=args.require_confidence,
            )
            if loaded and cx.has_confidence_data:
                metrics = cx.calculate_confidence_metrics()
                row['iptm'] = metrics.get('iptm', '')
                row['ptm'] = metrics.get('ptm', '')
                row['ipae'] = metrics.get('interface_pae', '')
                row['min_ipae'] = metrics.get('min_interface_pae', '')
                row['confidence'] = metrics.get('combined_confidence', '')

    except Exception as e:
        row['error'] = str(e)
        print(f'Error processing {filepath}: {e}', file=sys.stderr)

    return row


def main(argv=None):
    args = parse_args(argv)
    files = collect_files(args)

    print(f'Processing {len(files)} file(s)...', file=sys.stderr)

    if args.cpu == 1 or len(files) == 1:
        results = [process_one(f, args) for f in files]
    else:
        from joblib import Parallel, delayed
        n_jobs = args.cpu if args.cpu != -1 else -1
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_one)(f, args) for f in files
        )

    df = pd.DataFrame(results)
    df.to_csv(args.output, sep='\t', index=False)
    print(f'Wrote {len(df)} rows to {args.output}', file=sys.stderr)


if __name__ == '__main__':
    main()

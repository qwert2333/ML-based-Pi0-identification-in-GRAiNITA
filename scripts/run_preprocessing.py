#!/usr/bin/env python3
"""Master Preprocessing Executable.

Processes raw CLUE Ntuples into version-tagged tabular (BDT) or 
spatial tensor (GATr) datasets using the Dataset Catalog registry.
"""

import argparse
import glob
import os
import sys
from pathlib import Path

# Automatically ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.datasets import DATASET_CATALOG
from configs.paths import FilePaths
#from src.preprocessing.bdt_data_prep import run_bdt_preprocessing
from src.preprocessing.bdt_data_prep_select import run_bdt_preprocessing
from src.preprocessing.gatr_data_prep import run_gatr_preprocessing


def resolve_input_files(pattern_or_dir: str, base_dir: str, max_files: int) -> list[str]:
    """Expands a directory name, relative path, or glob pattern into sorted ROOT file paths."""
    candidate = (
        pattern_or_dir
        if os.path.isabs(pattern_or_dir)
        else os.path.join(base_dir, pattern_or_dir)
    )

    if os.path.isdir(candidate):
        search_path = os.path.join(candidate, "*.root")
    else:
        search_path = candidate

    matched_files = sorted(glob.glob(search_path))

    if not matched_files and os.path.isdir(candidate):
        recursive_path = os.path.join(candidate, "**", "*.root")
        matched_files = sorted(glob.glob(recursive_path, recursive=True))

    if not matched_files:
        raise FileNotFoundError(f"[ERROR] No ROOT files found matching: {search_path}")

    return matched_files[:max_files] if max_files > 0 else matched_files


def main():
    parser = argparse.ArgumentParser(
        description="Master CLI for CLUE dataset preprocessing."
    )

    # Core execution flags
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["bdt", "gatr", "all"],
        help="Target architecture pipeline to execute.",
    )
    parser.add_argument(
        "--dataset-key",
        type=str,
        default=None,
        choices=list(DATASET_CATALOG.keys()),
        help="Pre-configured dataset key from DATASET_CATALOG.",
    )
    parser.add_argument(
        "--pion-dir",
        type=str,
        default=None,
        help="Manual override for pion (bkg) directory/glob pattern.",
    )
    parser.add_argument(
        "--photon-dir",
        type=str,
        default=None,
        help="Manual override for photon (sig) directory/glob pattern.",
    )
    parser.add_argument(
        "--version-tag",
        type=str,
        default=None,
        help="Custom dataset version tag. Defaults to --dataset-key if omitted.",
    )

    # Parameters
    parser.add_argument(
        "--num-files",
        type=int,
        default=-1,
        help="Max ROOT files to process per class (-1 for all matched files).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for 50/50 balance and train/val/test split consistency.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="Events per chunk in exported split ROOT datasets.",
    )
    parser.add_argument(
        "--files-per-chunk",
        type=int,
        default=25,
        help="[GATr] Raw ROOT files combined per intermediate tensor chunk.",
    )
    parser.add_argument(
        "--merge-back-layers",
        action="store_true",
        help=(
            "Enable 1:3 hit readout: keep layer 0 and merge layers 1/2/3 "
            "per angular cell, summing energy at the layer-2 position."
        ),
    )
    parser.add_argument(
        "--merge-all-layers",
        action="store_true",
        help=(
            "[BDT] Enable 1-layer readout: merge layers 0/1/2/3 per angular "
            "cell, summing energy at the detector-middle position (between "
            "layers 1 and 2) along the projective cell direction."
        ),
    )
    parser.add_argument(
        "--output-base-dir",
        type=str,
        default=None,
        help="Optional output base containing BDT/ and GATr/ (defaults to FilePaths.DATA_DIR).",
    )

    args = parser.parse_args()
    if args.merge_back_layers and args.merge_all_layers:
        parser.error("--merge-back-layers and --merge-all-layers are exclusive.")
    if args.merge_all_layers and args.model != "bdt":
        parser.error("--merge-all-layers is only implemented for --model bdt.")

    # 1. Resolve Dataset Configuration
    if args.dataset_key:
        catalog_entry = DATASET_CATALOG[args.dataset_key]
        pion_pattern = args.pion_dir or catalog_entry["pion_dir"]
        photon_pattern = args.photon_dir or catalog_entry["photon_dir"]
        version_tag = args.version_tag or args.dataset_key
        print(f"--> Using Catalog Entry '{args.dataset_key}': {catalog_entry.get('description', '')}")
    elif args.pion_dir and args.photon_dir:
        pion_pattern = args.pion_dir
        photon_pattern = args.photon_dir
        version_tag = args.version_tag or "custom"
        print("--> Using custom CLI input directory paths.")
    else:
        parser.error("You must specify either --dataset-key OR both --pion-dir and --photon-dir.")

    # 2. Initialize Paths & Resolve Input Files
    paths = FilePaths()
    output_base_dir = (
        os.path.abspath(args.output_base_dir)
        if args.output_base_dir
        else paths.DATA_DIR
    )
    pion_inputs = resolve_input_files(pion_pattern, paths.RAW_DATA_IN, args.num_files)
    photon_inputs = resolve_input_files(photon_pattern, paths.RAW_DATA_IN, args.num_files)

    print(f"--> Resolved Inputs: {len(photon_inputs)} Photon (Signal) & {len(pion_inputs)} Pion (Background) files.")

    # 3. Execute BDT Pipeline
    if args.model in ["bdt", "all"]:
        bdt_target_dir = os.path.join(output_base_dir, "BDT", f"dataset_{version_tag}")
        os.makedirs(bdt_target_dir, exist_ok=True)

        print("\n" + "=" * 60)
        print(f"  RUNNING BDT PREPROCESSING [Tag: {version_tag}]")
        print(f"  Target Dir: {bdt_target_dir}")
        print("=" * 60)

        run_bdt_preprocessing(
            signal_files=pion_inputs,
            background_files=photon_inputs,
            output_dir=bdt_target_dir,
            seed=args.seed,
            chunk_size=args.chunk_size,
            merge_back_layers=args.merge_back_layers,
            merge_all_layers=args.merge_all_layers,
        )

    # 4. Execute GATr Pipeline
    if args.model in ["gatr", "all"]:
        gatr_target_dir = os.path.join(output_base_dir, "GATr", f"dataset_{version_tag}")
        os.makedirs(gatr_target_dir, exist_ok=True)

        print("\n" + "=" * 60)
        print(f"  RUNNING GATR PREPROCESSING [Tag: {version_tag}]")
        print(f"  Target Dir: {gatr_target_dir}")
        print("=" * 60)

        run_gatr_preprocessing(
            signal_files=pion_inputs,
            background_files=photon_inputs,
            output_dir=gatr_target_dir,
            seed=args.seed,
            chunk_size=args.chunk_size,
            files_per_chunk=args.files_per_chunk,
            version_tag=version_tag,
            merge_back_layers=args.merge_back_layers,
        )

    print("\n[SUCCESS] Pipeline processing completed successfully!")


if __name__ == "__main__":
    main()
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
    # Build absolute search path
    if not os.path.isabs(pattern_or_dir) and not ("*" in pattern_or_dir or "?" in pattern_or_dir):
        search_path = os.path.join(base_dir, pattern_or_dir, "*.root")
    elif not os.path.isabs(pattern_or_dir):
        search_path = os.path.join(base_dir, pattern_or_dir)
    else:
        search_path = pattern_or_dir

    matched_files = sorted(glob.glob(search_path))

    # If simple search yielded nothing, try recursive lookup inside the directory
    if not matched_files and os.path.isdir(os.path.join(base_dir, pattern_or_dir)):
        recursive_path = os.path.join(base_dir, pattern_or_dir, "**", "*.root")
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

    args = parser.parse_args()

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
    pion_inputs = resolve_input_files(pion_pattern, paths.RAW_DATA_IN, args.num_files)
    photon_inputs = resolve_input_files(photon_pattern, paths.RAW_DATA_IN, args.num_files)

    print(f"--> Resolved Inputs: {len(photon_inputs)} Photon (Signal) & {len(pion_inputs)} Pion (Background) files.")

    # 3. Execute BDT Pipeline
    if args.model in ["bdt", "all"]:
        bdt_target_dir = os.path.join(paths.BDT_DATA_DIR, f"dataset_{version_tag}")
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
        )

    # 4. Execute GATr Pipeline
    if args.model in ["gatr", "all"]:
        gatr_target_dir = os.path.join(paths.GATR_DATA_DIR, f"dataset_{version_tag}")
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
        )

    print("\n[SUCCESS] Pipeline processing completed successfully!")


if __name__ == "__main__":
    main()
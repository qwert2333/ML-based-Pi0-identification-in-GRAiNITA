"""Centralized File Path Manager."""

import glob
import os
from pathlib import Path


class FilePaths:
    """Manages file paths for raw data, preprocessed datasets, trained model checkpoints,

    and shared evaluation outputs across both GATr and BDT pipelines.
    """

    def __init__(self, base_dir=None, project_dir=None):
        # 1. Base & Project Root. Environment overrides make batch workers use
        # this checkout rather than the original author's EOS directory.
        local_project_dir = str(Path(__file__).resolve().parents[1])
        self.PROJECT_DIR = os.path.abspath(
            project_dir
            or os.environ.get("PI0ID_PROJECT_DIR", local_project_dir)
        )
        self.BASE_DIR = os.path.abspath(
            base_dir
            or os.environ.get("PI0ID_BASE_DIR", os.path.dirname(self.PROJECT_DIR))
        )

        # 2. Raw Input Directory
        self.RAW_DATA_IN = os.environ.get(
            "PI0ID_RAW_DATA_DIR", os.path.join(self.PROJECT_DIR, "data", "RawData")
        )

        # 3. Model-Specific Preprocessed Data Directories
        self.DATA_DIR = os.path.join(self.PROJECT_DIR, "data")
        self.BDT_DATA_DIR = os.environ.get(
            "PI0ID_BDT_DATA_DIR", os.path.join(self.DATA_DIR, "BDT")
        )
        self.GATR_DATA_DIR = os.environ.get(
            "PI0ID_GATR_DATA_DIR", os.path.join(self.DATA_DIR, "GATr")
        )

        # 4. Model Checkpoint Output Directories
        self.MODELS_DIR = os.path.join(self.PROJECT_DIR, "trained_models")
        self.BDT_MODELS_DIR = os.path.join(self.MODELS_DIR, "BDT")
        self.GATR_MODELS_DIR = os.path.join(self.MODELS_DIR, "GATr")

        # 5. Shared Evaluation & Results Directories
        self.RESULTS_DIR = os.path.join(self.PROJECT_DIR, "results")
        self.EVAL_PLOTS_DIR = os.path.join(self.RESULTS_DIR, "plots")
        self.EVAL_METRICS_DIR = os.path.join(self.RESULTS_DIR, "metrics")

        # 6. Standard Split Names
        self.SPLIT_NAMES = {
            "train": "Training",
            "val": "Validation",
            "test": "Testing",
        }

    def ensure_dirs(self):
        """Creates all output directories if they don't already exist."""
        output_dirs = [
            self.DATA_DIR,
            self.BDT_DATA_DIR,
            self.GATR_DATA_DIR,
            self.MODELS_DIR,
            self.BDT_MODELS_DIR,
            self.GATR_MODELS_DIR,
            self.RESULTS_DIR,
            self.EVAL_PLOTS_DIR,
            self.EVAL_METRICS_DIR,
        ]
        for d in output_dirs:
            os.makedirs(d, exist_ok=True)

    def get_split_dir(self, model_type: str, version_tag: str, split: str) -> str:
        """Returns the full path to a specific dataset split folder.

        Supports optional intermediate subfolders like /split/, /splits/, or /processed/.
        """
        model_data_dir = (
            self.GATR_DATA_DIR
            if model_type.lower() == "gatr"
            else self.BDT_DATA_DIR
        )

        split_key = split.lower()
        split_folder = self.SPLIT_NAMES.get(split_key, split)
        split_variations = list(dict.fromkeys([split_folder, split_key, split_key.capitalize(), split.upper()]))

        prefix_variations = [
            f"dataset_{version_tag}",
            f"split_{version_tag}",
            f"processed_{version_tag}",
            version_tag,
        ]

        intermediate_subfolders = ["", "split", "splits", "processed"]
        base_dirs = [model_data_dir, self.DATA_DIR]

        # Pass 1: Find non-empty directory containing actual files
        for bdir in base_dirs:
            for pref in prefix_variations:
                for inter in intermediate_subfolders:
                    for svar in split_variations:
                        candidate = (
                            os.path.join(bdir, pref, inter, svar)
                            if inter
                            else os.path.join(bdir, pref, svar)
                        )
                        if os.path.exists(candidate) and len(os.listdir(candidate)) > 0:
                            return candidate

        # Pass 2: Fallback to existing directory
        for bdir in base_dirs:
            for pref in prefix_variations:
                for inter in intermediate_subfolders:
                    for svar in split_variations:
                        candidate = (
                            os.path.join(bdir, pref, inter, svar)
                            if inter
                            else os.path.join(bdir, pref, svar)
                        )
                        if os.path.exists(candidate):
                            return candidate

        # Default fallback
        return os.path.join(model_data_dir, f"dataset_{version_tag}", "split", split_folder)

    def get_gatr_file_patterns(self, version_tag: str) -> dict:
        """Constructs glob pattern paths for GATr files per split."""
        patterns = {}
        for split in ["train", "val", "test"]:
            split_dir = self.get_split_dir("gatr", version_tag, split)

            possible_patterns = [
                os.path.join(split_dir, "clue_*_gatr_chunk_*.root"),
                os.path.join(split_dir, "*_chunk_*.root"),
                os.path.join(split_dir, "*.root"),
            ]

            matched_pattern = os.path.join(split_dir, "*.root")
            for pat in possible_patterns:
                if glob.glob(pat):
                    matched_pattern = pat
                    break

            patterns[split] = matched_pattern

        return patterns
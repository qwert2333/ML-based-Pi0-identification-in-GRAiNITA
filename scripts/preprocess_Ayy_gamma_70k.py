#!/usr/bin/env python3
"""Preprocess the completed 70k-event A->gamma gamma / single-gamma sample for GATr.

Processes one RecTuple per resumable intermediate file, then makes event-grouped
training, validation, and testing splits. No model training is performed.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import uproot

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.preprocessing.gatr_data_prep import (
    preprocess_clue_root_file,
    print_cutflow_table,
    process_mix_and_split,
)

INPUT_ROOT = PROJECT_ROOT / "data/dd4hep_Ayy_gamma_70k"
# Kept separate from the original 256-hit dataset_Ayy_gamma_70k.
OUTPUT_ROOT = PROJECT_ROOT / "data/GATr/dataset_Ayy_gamma_70k_h512"
MAX_HITS = 512
SPLIT_SEED = 42
SPLIT_CHUNK_SIZE = 5000
EXPECTED_FILES = {"signal": 35, "background": 19}


def nominal_tags(path: str) -> dict[str, str]:
    match = re.search(r"_mA([^_]+)_tau([^_]+)_", Path(path).name)
    if match is None:
        return {"nominal_mass": "", "nominal_lifetime": ""}
    return {
        "nominal_mass": f"mA{match.group(1)}",
        "nominal_lifetime": f"tau{match.group(2)}",
    }


def validate_intermediate(path: Path, source: Path) -> int:
    expected_id = zlib.crc32(str(source.resolve()).encode())
    with uproot.open(path) as root_file:
        main = root_file["CLUEShowers"]
        extra = root_file["CLUEExtra"]
        count = main.num_entries
        if count < 1 or extra.num_entries != count:
            raise ValueError(f"Invalid cluster count in {path}")
        source_ids = main["source_file_id"].array(library="np")
        if not np.all(source_ids == expected_id):
            raise ValueError(f"Source mismatch in {path}")
    return count


def audit_split(split_root: Path) -> dict:
    seen_events: dict[str, set[int]] = {}
    seen_sources: set[int] = set()
    counts: dict[str, dict] = {}
    for split in ("Training", "Validation", "Testing"):
        files = sorted((split_root / split).glob("*.root"))
        if not files:
            raise ValueError(f"No output chunks for {split}")
        event_ids: set[int] = set()
        label_counts = Counter()
        regression_count = 0
        cluster_count = 0
        for path in files:
            with uproot.open(path) as root_file:
                main = root_file["CLUEShowers"]
                extra = root_file["CLUEExtra"]
                if main.num_entries != extra.num_entries:
                    raise ValueError(f"Tree length mismatch: {path}")
                ids = main["event_uid"].array(library="np")
                event_ids.update(map(int, ids))
                seen_sources.update(map(int, np.unique(main["source_file_id"].array(library="np"))))
                label_counts.update(map(int, main["label"].array(library="np")))
                regression_count += int(main["regression_valid"].array(library="np").sum())
                cluster_count += main.num_entries
        seen_events[split] = event_ids
        counts[split] = {
            "files": len(files), "events": len(event_ids), "clusters": cluster_count,
            "label_1": label_counts[1], "label_0": label_counts[0],
            "label_invalid": label_counts[-1], "regression_valid": regression_count,
        }
    for i, first in enumerate(seen_events):
        for second in list(seen_events)[i + 1:]:
            if seen_events[first] & seen_events[second]:
                raise ValueError(f"Event leakage between {first} and {second}")
    if len(seen_sources) != sum(EXPECTED_FILES.values()):
        raise ValueError(f"Expected 54 input sources in splits; found {len(seen_sources)}")
    return counts


def main() -> None:
    inputs = {
        kind: sorted((INPUT_ROOT / kind).glob("**/RecTuple_*.root"))
        for kind in EXPECTED_FILES
    }
    for kind, expected in EXPECTED_FILES.items():
        if len(inputs[kind]) != expected:
            raise ValueError(f"Expected {expected} {kind} RecTuples; found {len(inputs[kind])}")
    processed = OUTPUT_ROOT / "intermediate_processed"
    processed.mkdir(parents=True, exist_ok=True)
    flows = {kind: Counter() for kind in EXPECTED_FILES}
    intermediates = {}
    for kind, files in inputs.items():
        intermediates[kind] = []
        for index, source in enumerate(files):
            target = processed / f"preprocessed_{kind}_{index:03d}.root"
            if target.exists():
                try:
                    count = validate_intermediate(target, source)
                    print(f"[{kind} {index + 1}/{len(files)}] reuse {count} clusters: {source.name}", flush=True)
                    intermediates[kind].append(target)
                    continue
                except Exception as exc:
                    raise ValueError(f"Existing intermediate failed validation: {target}: {exc}") from exc
            preprocess_clue_root_file(
                [str(source)], str(target), max_hits=MAX_HITS,
                cutflow_logger=flows[kind], llp_pdg_ids=(9000005,),
                sample_metadata_fn=nominal_tags,
            )
            count = validate_intermediate(target, source)
            print(f"[{kind} {index + 1}/{len(files)}] saved {count} clusters: {source.name}", flush=True)
            intermediates[kind].append(target)
    for kind in flows:
        print_cutflow_table(flows[kind], f"{kind} newly processed")
    split_tmp = OUTPUT_ROOT / "split_in_progress"
    if split_tmp.exists():
        shutil.rmtree(split_tmp)
    process_mix_and_split(
        intermediates["signal"], intermediates["background"], str(split_tmp),
        chunk_size=SPLIT_CHUNK_SIZE, random_seed=SPLIT_SEED,
    )
    counts = audit_split(split_tmp)
    final_split = OUTPUT_ROOT / "split"
    if final_split.exists():
        shutil.rmtree(final_split)
    split_tmp.rename(final_split)
    summary = {
        "input_root": str(INPUT_ROOT), "output_root": str(OUTPUT_ROOT),
        "signal_files": len(inputs["signal"]),
        "background_files": len(inputs["background"]),
        "max_hits": MAX_HITS, "split_seed": SPLIT_SEED,
        "split_chunk_size": SPLIT_CHUNK_SIZE, "splits": counts,
    }
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

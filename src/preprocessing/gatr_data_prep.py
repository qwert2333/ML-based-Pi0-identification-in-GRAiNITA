"""Cluster-level preprocessing for GATr LLP reconstruction.

One output row represents one CLUE cluster. Truth units are GeV, mm, and ns.
Hit coordinates are kept physical and normalized inside the model.
"""
from collections import Counter
from itertools import combinations
import os, shutil, tempfile, zlib
import awkward as ak
import numpy as np
from sklearn.model_selection import train_test_split
import uproot

MAX_HITS_DEFAULT = 256
# Front (thin) longitudinal layer; its energy share is the only longitudinal
# information available, since the rear layer holds nearly all the energy.
FIRST_LAYER_ID = 0
DEFAULT_LLP_PDGS = (111, 9000005)
HIT_MAIN_KEYS = ["hit_E_raw", "hit_time_raw", "hit_layer", "hit_mask"]
EXTRA_KEYS = ["hit_x", "hit_y", "hit_z", "hit_cluster_id"]
STRING_METADATA_KEYS = ["nominal_mass", "nominal_lifetime"]
METADATA_DTYPES = {
    "source_file_id": np.uint32, "source_event_index": np.int64,
    "event_number": np.int64, "event_uid": np.uint64,
    "cluster_id": np.int32, "cluster_rank": np.int32,
    "n_clusters_event": np.int32, "n_hits": np.int32,
    "n_hits_original": np.int32, "cluster_energy": np.float32,
    "cluster_energy_kept": np.float32,
    "cluster_energy_fraction_kept": np.float32,
    "cluster_first_layer_energy_fraction": np.float32,
    "cluster_x": np.float32, "cluster_y": np.float32,
    "cluster_z": np.float32, "cluster_time_raw": np.float32,
    "label": np.int32, "classification_valid": np.bool_,
    "event_label": np.int32, "regression_valid": np.bool_,
    "llp_truth_valid": np.bool_, "mass_valid": np.bool_,
    "decay_vertex_valid": np.bool_, "llp_pdg": np.int32,
    "llp_mass_GeV": np.float32, "llp_energy_GeV": np.float32,
    "llp_px_GeV": np.float32, "llp_py_GeV": np.float32,
    "llp_pz_GeV": np.float32, "llp_pt_GeV": np.float32,
    "llp_eta": np.float32, "llp_phi": np.float32,
    "llp_beta": np.float32, "llp_gamma": np.float32,
    "llp_production_x_mm": np.float32, "llp_production_y_mm": np.float32,
    "llp_production_z_mm": np.float32, "llp_production_time_ns": np.float32,
    "llp_decay_x_mm": np.float32, "llp_decay_y_mm": np.float32,
    "llp_decay_z_mm": np.float32, "llp_decay_time_ns": np.float32,
    "llp_decay_dx_mm": np.float32, "llp_decay_dy_mm": np.float32,
    "llp_decay_dz_mm": np.float32, "llp_decay_length_mm": np.float32,
    "llp_decay_radius_mm": np.float32, "llp_proper_time_ns": np.float32,
    "daughter_count": np.int32, "daughter1_pdg": np.int32,
    "daughter1_energy_GeV": np.float32, "daughter1_px_GeV": np.float32,
    "daughter1_py_GeV": np.float32, "daughter1_pz_GeV": np.float32,
    "daughter2_pdg": np.int32, "daughter2_energy_GeV": np.float32,
    "daughter2_px_GeV": np.float32, "daughter2_py_GeV": np.float32,
    "daughter2_pz_GeV": np.float32,
    "daughter_opening_angle_rad": np.float32,
    "daughter_energy_asymmetry": np.float32,
    "daughter_pair_mass_GeV": np.float32,
}
MAIN_KEYS = HIT_MAIN_KEYS + list(METADATA_DTYPES) + STRING_METADATA_KEYS

def print_cutflow_table(cutflow: Counter, label: str):
    print(f"\n================ {label} ================")
    for key in sorted(cutflow):
        print(f"{key:<42} {cutflow[key]:>10}")
    print("=" * 56)

def _mass(energy, momentum):
    return float(np.sqrt(max(float(energy)**2 - float(np.dot(momentum, momentum)), 0.0)))

def _eta(momentum):
    p, pz = float(np.linalg.norm(momentum)), float(momentum[2])
    return float(np.copysign(np.inf, pz)) if p <= abs(pz) else float(
        0.5 * np.log((p + pz) / (p - pz))
    )

def _empty_truth():
    out = {}
    for key, dtype in METADATA_DTYPES.items():
        out[key] = np.float32(np.nan) if np.issubdtype(dtype, np.floating) else dtype(0)
    out.update({
        "label": np.int32(-1), "classification_valid": np.bool_(False),
        "event_label": np.int32(0), "regression_valid": np.bool_(False),
        "llp_truth_valid": np.bool_(False), "mass_valid": np.bool_(False),
        "decay_vertex_valid": np.bool_(False), "llp_pdg": np.int32(0),
        "daughter1_pdg": np.int32(0), "daughter2_pdg": np.int32(0),
    })
    return out

def _best_photon_pair(pdg, vertices, momenta, energies, parent_p, parent_e):
    photons = np.flatnonzero((pdg == 22) & np.isfinite(energies) & (energies > 0))
    if len(photons) < 2:
        return None
    pscale, escale = max(np.linalg.norm(parent_p), 1e-6), max(abs(parent_e), 1e-6)
    ranked = []
    for first, second in combinations(photons, 2):
        vertex_gap = np.linalg.norm(vertices[first] - vertices[second])
        closure = (
            np.linalg.norm(momenta[first] + momenta[second] - parent_p) / pscale
            + abs(energies[first] + energies[second] - parent_e) / escale
        )
        ranked.append((vertex_gap + 1000.0 * closure, first, second, vertex_gap, closure))
    _, first, second, vertex_gap, closure = min(ranked)
    return (int(first), int(second)) if vertex_gap < 1e-3 and closure < 1e-3 else None

def extract_event_truth(mc_event, llp_pdg_ids=DEFAULT_LLP_PDGS):
    """Extract LLP four-vector, production/decay vertices, and two-photon truth."""
    out = _empty_truth()
    pdg = np.asarray(mc_event["pdg"], np.int32)
    energy = np.asarray(mc_event["energy"], np.float64)
    primary = np.asarray(mc_event["primary"], np.bool_)
    px = np.asarray(mc_event["p_x"], np.float64)
    py = np.asarray(mc_event["p_y"], np.float64)
    pz = np.asarray(mc_event["p_z"], np.float64)
    momentum = np.column_stack((px, py, pz))
    vertex = np.column_stack(tuple(np.asarray(mc_event[k], np.float64)
        for k in ("vertex_x", "vertex_y", "vertex_z")))
    time = np.asarray(mc_event["time"], np.float64)
    events = np.asarray(mc_event["event"], np.int64)
    out["event_number"] = np.int64(events[0] if len(events) else -1)
    candidates = np.flatnonzero(np.isin(pdg, np.asarray(llp_pdg_ids, np.int32)))
    if not len(candidates):
        photons = np.flatnonzero((pdg == 22) & primary)
        if len(photons):
            idx = int(photons[np.argmax(energy[photons])])
            out.update({
                "llp_pdg": np.int32(pdg[idx]),
                "llp_mass_GeV": np.float32(_mass(energy[idx], momentum[idx])),
                "llp_energy_GeV": np.float32(energy[idx]),
                "llp_px_GeV": np.float32(px[idx]), "llp_py_GeV": np.float32(py[idx]),
                "llp_pz_GeV": np.float32(pz[idx]),
                "llp_pt_GeV": np.float32(np.hypot(px[idx], py[idx])),
                "llp_eta": np.float32(_eta(momentum[idx])),
                "llp_phi": np.float32(np.arctan2(py[idx], px[idx])),
            })
        return out
    idx = int(candidates[np.argmax(energy[candidates])])
    parent_e, parent_p = float(energy[idx]), momentum[idx]
    parent_m, production = _mass(parent_e, parent_p), vertex[idx]
    pabs = float(np.linalg.norm(parent_p))
    out.update({
        "event_label": np.int32(1), "llp_truth_valid": np.bool_(True),
        "mass_valid": np.bool_(np.isfinite(parent_m) and parent_m > 0),
        "llp_pdg": np.int32(pdg[idx]), "llp_mass_GeV": np.float32(parent_m),
        "llp_energy_GeV": np.float32(parent_e),
        "llp_px_GeV": np.float32(parent_p[0]), "llp_py_GeV": np.float32(parent_p[1]),
        "llp_pz_GeV": np.float32(parent_p[2]),
        "llp_pt_GeV": np.float32(np.hypot(parent_p[0], parent_p[1])),
        "llp_eta": np.float32(_eta(parent_p)),
        "llp_phi": np.float32(np.arctan2(parent_p[1], parent_p[0])),
        "llp_beta": np.float32(pabs / parent_e if parent_e > 0 else np.nan),
        "llp_gamma": np.float32(parent_e / parent_m if parent_m > 0 else np.nan),
        "llp_production_x_mm": np.float32(production[0]),
        "llp_production_y_mm": np.float32(production[1]),
        "llp_production_z_mm": np.float32(production[2]),
        "llp_production_time_ns": np.float32(time[idx]),
    })
    pair = _best_photon_pair(pdg, vertex, momentum, energy, parent_p, parent_e)
    if pair is None:
        return out
    first, second = pair
    if energy[second] > energy[first]:
        first, second = second, first
    decay = (vertex[first] + vertex[second]) / 2
    decay_time = float((time[first] + time[second]) / 2)
    displacement = decay - production
    pair_e, pair_p = float(energy[first] + energy[second]), momentum[first] + momentum[second]
    denom = np.linalg.norm(momentum[first]) * np.linalg.norm(momentum[second])
    cos_angle = np.clip(np.dot(momentum[first], momentum[second]) / denom, -1, 1)
    proper_time = (
        (decay_time - time[idx]) * parent_m / parent_e
        if parent_e > 0 and parent_m > 0 else np.nan
    )
    out.update({
        "decay_vertex_valid": np.bool_(np.all(np.isfinite(decay))),
        "llp_decay_x_mm": np.float32(decay[0]), "llp_decay_y_mm": np.float32(decay[1]),
        "llp_decay_z_mm": np.float32(decay[2]),
        "llp_decay_time_ns": np.float32(decay_time),
        "llp_decay_dx_mm": np.float32(displacement[0]),
        "llp_decay_dy_mm": np.float32(displacement[1]),
        "llp_decay_dz_mm": np.float32(displacement[2]),
        "llp_decay_length_mm": np.float32(np.linalg.norm(displacement)),
        "llp_decay_radius_mm": np.float32(np.hypot(displacement[0], displacement[1])),
        "llp_proper_time_ns": np.float32(proper_time), "daughter_count": np.int32(2),
        "daughter1_pdg": np.int32(pdg[first]),
        "daughter1_energy_GeV": np.float32(energy[first]),
        "daughter1_px_GeV": np.float32(px[first]),
        "daughter1_py_GeV": np.float32(py[first]),
        "daughter1_pz_GeV": np.float32(pz[first]),
        "daughter2_pdg": np.int32(pdg[second]),
        "daughter2_energy_GeV": np.float32(energy[second]),
        "daughter2_px_GeV": np.float32(px[second]),
        "daughter2_py_GeV": np.float32(py[second]),
        "daughter2_pz_GeV": np.float32(pz[second]),
        "daughter_opening_angle_rad": np.float32(np.arccos(cos_angle)),
        "daughter_energy_asymmetry": np.float32(
            abs(energy[first] - energy[second]) / pair_e if pair_e else np.nan),
        "daughter_pair_mass_GeV": np.float32(_mass(pair_e, pair_p)),
    })
    return out

def _event_uid(path, event_index):
    file_id = np.uint32(zlib.crc32(os.path.abspath(path).encode()))
    return file_id, np.uint64((int(file_id) << 32) | (event_index & 0xFFFFFFFF))

def preprocess_clue_root_file(
    input_root_paths, output_root_path, stats=None, max_hits=MAX_HITS_DEFAULT,
    extra_trees_to_load=None, preselection_fn=None, cutflow_logger=None,
    llp_pdg_ids=DEFAULT_LLP_PDGS, nominal_mass="", nominal_lifetime="",
    sample_metadata_fn=None,
):
    """Convert RecTuple inputs into fixed-size cluster rows."""
    del stats, extra_trees_to_load
    cutflow = cutflow_logger if cutflow_logger is not None else Counter()
    main, extra = ({key: [] for key in MAIN_KEYS}, {key: [] for key in EXTRA_KEYS})
    valid_files = 0
    mc_keys = ["event", "pdg", "energy", "primary", "vertex_x", "vertex_y",
               "vertex_z", "p_x", "p_y", "p_z", "time"]
    hit_keys = ["x", "y", "z", "layer", "time", "energy"]
    for path in input_root_paths:
        try:
            path_metadata = {
                "nominal_mass": nominal_mass(path) if callable(nominal_mass) else nominal_mass,
                "nominal_lifetime": (
                    nominal_lifetime(path) if callable(nominal_lifetime) else nominal_lifetime
                ),
            }
            if sample_metadata_fn is not None:
                path_metadata.update(sample_metadata_fn(path) or {})
            path_metadata = {
                key: str(path_metadata.get(key, "") or "") for key in STRING_METADATA_KEYS
            }
            with uproot.open(path) as root:
                if "CLUEClustersHits" not in root or "MCParticles" not in root:
                    raise KeyError("missing CLUEClustersHits or MCParticles")
                hit_tree = root["CLUEClustersHits"]
                cid_key = next((k for k in ("hit_cluster_id", "cluster_id", "clusterId", "cluster")
                                if k in hit_tree.keys()), None)
                if cid_key is None:
                    raise KeyError("no cluster ID branch")
                hits = hit_tree.arrays(hit_keys + [cid_key], library="ak")
                mc = root["MCParticles"].arrays(mc_keys, library="ak")
                if len(hits) != len(mc):
                    raise ValueError("hit and MC tree entry counts differ")
                selected = np.ones(len(mc), dtype=bool)
                if preselection_fn is not None:
                    clusters = root["CLUEClusters"].arrays(library="ak")
                    selected = np.asarray(ak.to_numpy(preselection_fn({
                        "CLUEClustersHits": hits, "CLUEClusters": clusters,
                        "MCParticles": mc}, cutflow)), dtype=bool)
                for event_index in np.flatnonzero(selected):
                    cutflow["0_events_seen"] += 1
                    truth = extract_event_truth(
                        {k: ak.to_numpy(mc[k][event_index]) for k in mc_keys},
                        llp_pdg_ids,
                    )
                    hx, hy, hz = (ak.to_numpy(hits[k][event_index]).astype(np.float32)
                                  for k in ("x", "y", "z"))
                    he = ak.to_numpy(hits["energy"][event_index]).astype(np.float32)
                    ht = ak.to_numpy(hits["time"][event_index]).astype(np.float32)
                    hl = ak.to_numpy(hits["layer"][event_index]).astype(np.int32)
                    hc = ak.to_numpy(hits[cid_key][event_index]).astype(np.int32)
                    finite = (np.isfinite(hx) & np.isfinite(hy) & np.isfinite(hz)
                              & np.isfinite(he) & (he > 0) & (hc >= 0))
                    ids = np.unique(hc[finite])
                    if not len(ids):
                        cutflow["1_events_without_clusters"] += 1
                        continue
                    energies = {int(cid): float(he[finite & (hc == cid)].sum()) for cid in ids}
                    ids = sorted((int(x) for x in ids), key=energies.get, reverse=True)
                    file_id, uid = _event_uid(path, int(event_index))
                    for rank, cid in enumerate(ids):
                        select = finite & (hc == cid)
                        indices = np.flatnonzero(select)
                        original_n, original_e = len(indices), float(he[indices].sum())
                        # Computed before truncation: front-layer hits are soft
                        # and are the first to be dropped by the energy ranking.
                        first_layer_e = float(he[select & (hl == FIRST_LAYER_ID)].sum())
                        if original_n > max_hits:
                            indices = indices[np.argsort(he[indices])[::-1][:max_hits]]
                            cutflow["2_clusters_truncated"] += 1
                        kept_n, kept_e = len(indices), float(he[indices].sum())
                        bx, by, bz, be, bt = (np.zeros(max_hits, np.float32) for _ in range(5))
                        bl = np.zeros(max_hits, np.int32)
                        bm, bc = np.zeros(max_hits, bool), np.full(max_hits, -1, np.int32)
                        bx[:kept_n], by[:kept_n], bz[:kept_n] = hx[indices], hy[indices], hz[indices]
                        be[:kept_n], bt[:kept_n], bl[:kept_n] = he[indices], ht[indices], hl[indices]
                        bm[:kept_n], bc[:kept_n] = True, cid
                        weights = he[select]
                        center = np.average(np.column_stack((hx[select], hy[select], hz[select])),
                                            axis=0, weights=weights)
                        good_time = select & np.isfinite(ht) & (ht >= 0)
                        cluster_time = (np.average(ht[good_time], weights=he[good_time])
                                        if good_time.any() else np.nan)
                        is_llp = bool(truth["event_label"])
                        class_valid = (not is_llp) or len(ids) == 1
                        label = 1 if is_llp and len(ids) == 1 else 0
                        regression_valid = bool(is_llp and len(ids) == 1
                            and truth["mass_valid"] and truth["decay_vertex_valid"])
                        row = dict(truth)
                        row.update({
                            "source_file_id": file_id, "source_event_index": np.int64(event_index),
                            "event_uid": uid, "cluster_id": np.int32(cid),
                            "cluster_rank": np.int32(rank), "n_clusters_event": np.int32(len(ids)),
                            "n_hits": np.int32(kept_n), "n_hits_original": np.int32(original_n),
                            "cluster_energy": np.float32(original_e),
                            "cluster_energy_kept": np.float32(kept_e),
                            "cluster_energy_fraction_kept": np.float32(
                                kept_e / original_e if original_e else 0),
                            "cluster_first_layer_energy_fraction": np.float32(
                                first_layer_e / original_e if original_e else 0),
                            "cluster_x": np.float32(center[0]), "cluster_y": np.float32(center[1]),
                            "cluster_z": np.float32(center[2]),
                            "cluster_time_raw": np.float32(cluster_time),
                            "label": np.int32(label if class_valid else -1),
                            "classification_valid": np.bool_(class_valid),
                            "regression_valid": np.bool_(regression_valid),
                        })
                        main["hit_E_raw"].append(be); main["hit_time_raw"].append(bt)
                        main["hit_layer"].append(bl); main["hit_mask"].append(bm)
                        for key in METADATA_DTYPES: main[key].append(row[key])
                        for key in STRING_METADATA_KEYS:
                            main[key].append(path_metadata[key])
                        extra["hit_x"].append(bx); extra["hit_y"].append(by)
                        extra["hit_z"].append(bz); extra["hit_cluster_id"].append(bc)
                        cutflow["3_clusters_written"] += 1
                        cutflow["4_classification_targets"] += int(class_valid)
                        cutflow["5_regression_targets"] += int(regression_valid)
                valid_files += 1
        except Exception as error:
            print(f"Skipping {path}: {error}")
    if not main["label"]:
        raise ValueError(f"No valid clusters produced for {output_root_path}")
    main_out = {
        "hit_E_raw": np.asarray(main["hit_E_raw"], np.float32),
        "hit_time_raw": np.asarray(main["hit_time_raw"], np.float32),
        "hit_layer": np.asarray(main["hit_layer"], np.int32),
        "hit_mask": np.asarray(main["hit_mask"], bool),
    }
    for key, dtype in METADATA_DTYPES.items(): main_out[key] = np.asarray(main[key], dtype)
    for key in STRING_METADATA_KEYS:
        main_out[key] = ak.Array(main[key])
    extra_out = {key: np.asarray(extra[key],
        np.int32 if key == "hit_cluster_id" else np.float32) for key in EXTRA_KEYS}
    descriptor, tmp = tempfile.mkstemp(suffix=".root", dir=os.environ.get("TMPDIR", "/tmp"))
    os.close(descriptor)
    try:
        with uproot.recreate(tmp) as output:
            output["CLUEShowers"] = main_out; output["CLUEExtra"] = extra_out
        parent = os.path.dirname(output_root_path)
        if parent: os.makedirs(parent, exist_ok=True)
        shutil.move(tmp, output_root_path)
    finally:
        if os.path.exists(tmp): os.remove(tmp)
    print(f"Saved {len(main['label'])} clusters from {valid_files} files -> {output_root_path}")
    return output_root_path

def load_clue_data_from_files(files):
    combined = {k: [] for k in MAIN_KEYS + EXTRA_KEYS}
    for path in files:
        if not path or not os.path.exists(path):
            continue
        with uproot.open(path) as root:
            for key in MAIN_KEYS:
                library = "ak" if key in STRING_METADATA_KEYS else "np"
                combined[key].append(root["CLUEShowers"][key].array(library=library))
            for key in EXTRA_KEYS:
                combined[key].append(root["CLUEExtra"][key].array(library="np"))
    output = {}
    for key, values in combined.items():
        if key in STRING_METADATA_KEYS:
            output[key] = ak.concatenate(values) if values else ak.Array([])
        else:
            output[key] = np.concatenate(values) if values else np.asarray([])
    return output

def _group_split(uids, labels, train_size, val_size, test_size, seed):
    groups, first = np.unique(uids, return_index=True)
    group_labels = labels[first]
    def stratify(values):
        _, counts = np.unique(values, return_counts=True)
        return values if len(counts) > 1 and counts.min() >= 2 else None
    train, remainder = train_test_split(groups, train_size=train_size/(train_size+val_size+test_size),
        random_state=seed, stratify=stratify(group_labels))
    rem_labels = group_labels[np.searchsorted(groups, remainder)]
    val, test = train_test_split(remainder, train_size=val_size/(val_size+test_size),
        random_state=seed, stratify=stratify(rem_labels))
    return train, val, test

def process_mix_and_split(signal_files, background_files, base_output_dir,
    tree_name="CLUEShowers", extra_tree_name="CLUEExtra", chunk_size=5000,
    train_size=0.6, val_size=0.2, test_size=0.2, random_seed=42):
    """Split by event UID so clusters from one event never cross data splits."""
    data = load_clue_data_from_files(list(signal_files) + list(background_files))
    if not len(data["label"]): raise ValueError("No clusters available for splitting")
    groups = _group_split(data["event_uid"], data["event_label"],
                          train_size, val_size, test_size, random_seed)
    for name, selected_groups, prefix in zip(
        ("Training", "Validation", "Testing"), groups,
        ("clue_train_gatr", "clue_val_gatr", "clue_test_gatr")):
        rows = np.flatnonzero(np.isin(data["event_uid"], selected_groups))
        np.random.default_rng(random_seed).shuffle(rows)
        directory = os.path.join(base_output_dir, name); os.makedirs(directory, exist_ok=True)
        for chunk, start in enumerate(range(0, len(rows), chunk_size)):
            idx = rows[start:start+chunk_size]
            with uproot.recreate(os.path.join(directory, f"{prefix}_chunk_{chunk}.root")) as output:
                output[tree_name] = {k: data[k][idx] for k in MAIN_KEYS}
                output[extra_tree_name] = {k: data[k][idx] for k in EXTRA_KEYS}
        print(f"{name}: {len(selected_groups)} events, {len(rows)} clusters")

def run_gatr_preprocessing(signal_files, background_files, output_dir, seed=42,
    chunk_size=5000, files_per_chunk=25, version_tag="v1", preselection_fn=None,
    extra_trees=None, max_hits=MAX_HITS_DEFAULT, llp_pdg_ids=DEFAULT_LLP_PDGS,
    nominal_mass="", nominal_lifetime="", sample_metadata_fn=None):
    """Run cluster preprocessing and event-grouped train/validation/test splitting."""
    del version_tag, extra_trees
    processed_dir = os.path.join(output_dir, "intermediate_processed")
    split_dir = os.path.join(output_dir, "split")
    os.makedirs(processed_dir, exist_ok=True); os.makedirs(split_dir, exist_ok=True)
    flows = {"signal": Counter(), "background": Counter()}
    def process(files, name):
        outputs = []
        for chunk, start in enumerate(range(0, len(files), files_per_chunk)):
            path = os.path.join(processed_dir, f"preprocessed_{name}_{chunk}.root")
            preprocess_clue_root_file(files[start:start+files_per_chunk], path,
                max_hits=max_hits, preselection_fn=preselection_fn,
                cutflow_logger=flows[name], llp_pdg_ids=llp_pdg_ids,
                nominal_mass=nominal_mass, nominal_lifetime=nominal_lifetime,
                sample_metadata_fn=sample_metadata_fn)
            outputs.append(path)
        return outputs
    signal, background = process(signal_files, "signal"), process(background_files, "background")
    print_cutflow_table(flows["signal"], "Signal preprocessing")
    print_cutflow_table(flows["background"], "Background preprocessing")
    process_mix_and_split(signal, background, split_dir,
                          chunk_size=chunk_size, random_seed=seed)
    print(f"\nGATr preprocessing complete: {split_dir}")

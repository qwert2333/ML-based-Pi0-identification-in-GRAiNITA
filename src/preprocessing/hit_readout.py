"""Hit-level transformations shared by the BDT and GATr preprocessors."""

from __future__ import annotations

import awkward as ak
import numpy as np


def merge_four_layer_hits_to_one_plus_three(
    hits: ak.Array,
    *,
    front_layer: int = 0,
    back_layers: tuple[int, int, int] = (1, 2, 3),
    middle_layer: int = 2,
    theta_bin_size: float = 0.0032,
    phi_bins: int = 1975,
) -> ak.Array:
    """Convert four longitudinal hit layers into a 1:3 readout.

    Layer ``front_layer`` is retained hit-for-hit. Hits in ``back_layers`` are
    grouped by their common angular calorimeter cell. Each merged hit stores
    the sum of the three layer energies and is assigned output layer 1. Its
    position (and other metadata, such as cluster ID or time) is copied from
    the hit in ``middle_layer``.

    CLUE output is sparse, so a cell can carry energy in an outer back layer
    without having a stored middle-layer hit. In that case the middle-layer
    position is reconstructed without dropping energy: opposing outer-layer
    positions are interpolated when both exist; otherwise the available
    position is projected onto the event's middle-layer radial surface.

    The default angular segmentation matches GRAiNITA's
    FCCSWGridRhoPhiTheta readout (delta-theta=0.0032 and 1975 phi bins).
    """
    required_fields = {"x", "y", "z", "layer", "energy"}
    missing = required_fields.difference(hits.fields)
    if missing:
        raise ValueError(
            "Cannot build 1:3 hit readout; missing fields: "
            + ", ".join(sorted(missing))
        )
    if middle_layer not in back_layers:
        raise ValueError("middle_layer must be one of back_layers")
    if theta_bin_size <= 0 or phi_bins <= 0:
        raise ValueError("theta_bin_size and phi_bins must be positive")

    fields = list(hits.fields)
    merged_fields: dict[str, list[np.ndarray]] = {field: [] for field in fields}
    back_layers_array = np.asarray(back_layers, dtype=np.int64)
    phi_bin_size = 2.0 * np.pi / float(phi_bins)

    for event in hits:
        values = {field: np.asarray(ak.to_list(event[field])) for field in fields}
        if len(values["energy"]) == 0:
            for field in fields:
                merged_fields[field].append(values[field])
            continue

        layer = values["layer"].astype(np.int64, copy=False)
        is_back = np.isin(layer, back_layers_array)
        keep_indices = np.flatnonzero(~is_back)

        # Preserve the front layer (and any unexpected non-back layer) exactly.
        event_output = {
            field: values[field][keep_indices].tolist() for field in fields
        }

        x = values["x"].astype(np.float64, copy=False)
        y = values["y"].astype(np.float64, copy=False)
        z = values["z"].astype(np.float64, copy=False)
        radius = np.sqrt(x * x + y * y + z * z)
        safe_radius = np.where(radius > 0.0, radius, 1.0)
        theta = np.arccos(np.clip(z / safe_radius, -1.0, 1.0))
        phi = np.mod(np.arctan2(y, x), 2.0 * np.pi)

        theta_index = np.rint(theta / theta_bin_size).astype(np.int64)
        phi_index = np.rint(phi / phi_bin_size).astype(np.int64) % phi_bins

        groups: dict[tuple[int, int], list[int]] = {}
        for idx in np.flatnonzero(is_back):
            key = (int(theta_index[idx]), int(phi_index[idx]))
            groups.setdefault(key, []).append(int(idx))

        cylindrical_radius = np.hypot(x, y)
        middle_mask = layer == middle_layer
        target_cyl_radius = (
            float(np.median(cylindrical_radius[middle_mask]))
            if np.any(middle_mask)
            else None
        )

        for key in sorted(groups):
            group_indices = np.asarray(groups[key], dtype=np.int64)
            group_layers = layer[group_indices]
            middle_indices = group_indices[group_layers == middle_layer]

            if len(middle_indices):
                representative = int(middle_indices[0])
                merged_position = np.array(
                    [x[representative], y[representative], z[representative]],
                    dtype=np.float64,
                )
            else:
                inner = group_indices[group_layers < middle_layer]
                outer = group_indices[group_layers > middle_layer]
                if len(inner) and len(outer):
                    # The longitudinal sampling surfaces are equally spaced.
                    inner_pos = np.mean(
                        np.column_stack((x[inner], y[inner], z[inner])), axis=0
                    )
                    outer_pos = np.mean(
                        np.column_stack((x[outer], y[outer], z[outer])), axis=0
                    )
                    merged_position = 0.5 * (inner_pos + outer_pos)
                    representative = int(
                        group_indices[np.argmin(np.abs(group_layers - middle_layer))]
                    )
                else:
                    representative = int(
                        group_indices[np.argmin(np.abs(group_layers - middle_layer))]
                    )
                    merged_position = np.array(
                        [x[representative], y[representative], z[representative]],
                        dtype=np.float64,
                    )
                    source_cyl_radius = cylindrical_radius[representative]
                    if target_cyl_radius is not None and source_cyl_radius > 0.0:
                        merged_position *= target_cyl_radius / source_cyl_radius

            for field in fields:
                if field == "energy":
                    value = np.sum(values[field][group_indices], dtype=np.float64)
                elif field == "layer":
                    value = 1
                elif field == "x":
                    value = merged_position[0]
                elif field == "y":
                    value = merged_position[1]
                elif field == "z":
                    value = merged_position[2]
                else:
                    value = values[field][representative]
                event_output[field].append(value)

        for field in fields:
            merged_fields[field].append(
                np.asarray(event_output[field], dtype=values[field].dtype)
            )

    return ak.zip({field: ak.Array(events) for field, events in merged_fields.items()})


def merge_all_layer_hits_to_one(
    hits: ak.Array,
    *,
    layers: tuple[int, ...] = (0, 1, 2, 3),
    output_layer: int = 0,
    middle_cyl_radius_mm: float = 2416.5,
    theta_bin_size: float = 0.0032,
    phi_bins: int = 1975,
) -> ak.Array:
    """Convert four longitudinal hit layers into a single-layer readout.

    Hits in ``layers`` are grouped by their common angular calorimeter cell
    (same segmentation as ``merge_four_layer_hits_to_one_plus_three``). Each
    merged hit stores the summed energy and is assigned ``output_layer``.

    The merged hit sits at the middle of the full detector depth: on the
    cell's line of sight at cylindrical radius ``middle_cyl_radius_mm``. The
    GRAiNITA barrel layers lie at cylindrical radii of about 2261.6, 2364.9,
    2468.2 and 2571.4 mm and the cells are projective (hits of one cell in
    different layers lie on a line through the origin), so the default is the
    midpoint between layers 1 and 2 (equivalently layers 0 and 3). Whichever
    layers of the cell carry signal, a hit of the cell is scaled along that
    line to the middle radius; when layers 1 and 2 are both present this
    equals the midpoint of their positions. No energy weighting is used.
    The hit closest in layer to the middle provides the direction and the
    other fields (such as cluster ID or time), preferring higher energy on
    ties. Hits in layers outside ``layers`` are kept unchanged.
    """
    required_fields = {"x", "y", "z", "layer", "energy"}
    missing = required_fields.difference(hits.fields)
    if missing:
        raise ValueError(
            "Cannot build single-layer hit readout; missing fields: "
            + ", ".join(sorted(missing))
        )
    if theta_bin_size <= 0 or phi_bins <= 0:
        raise ValueError("theta_bin_size and phi_bins must be positive")
    if middle_cyl_radius_mm <= 0:
        raise ValueError("middle_cyl_radius_mm must be positive")

    fields = list(hits.fields)
    merged_fields: dict[str, list[np.ndarray]] = {field: [] for field in fields}
    layers_array = np.asarray(layers, dtype=np.int64)
    middle_layer = 0.5 * (float(layers_array.min()) + float(layers_array.max()))
    phi_bin_size = 2.0 * np.pi / float(phi_bins)

    for event in hits:
        values = {field: np.asarray(ak.to_list(event[field])) for field in fields}
        if len(values["energy"]) == 0:
            for field in fields:
                merged_fields[field].append(values[field])
            continue

        layer = values["layer"].astype(np.int64, copy=False)
        is_merged = np.isin(layer, layers_array)
        keep_indices = np.flatnonzero(~is_merged)
        event_output = {
            field: values[field][keep_indices].tolist() for field in fields
        }

        x = values["x"].astype(np.float64, copy=False)
        y = values["y"].astype(np.float64, copy=False)
        z = values["z"].astype(np.float64, copy=False)
        energy = values["energy"].astype(np.float64, copy=False)
        radius = np.sqrt(x * x + y * y + z * z)
        safe_radius = np.where(radius > 0.0, radius, 1.0)
        theta = np.arccos(np.clip(z / safe_radius, -1.0, 1.0))
        phi = np.mod(np.arctan2(y, x), 2.0 * np.pi)
        theta_index = np.rint(theta / theta_bin_size).astype(np.int64)
        phi_index = np.rint(phi / phi_bin_size).astype(np.int64) % phi_bins

        groups: dict[tuple[int, int], list[int]] = {}
        for idx in np.flatnonzero(is_merged):
            key = (int(theta_index[idx]), int(phi_index[idx]))
            groups.setdefault(key, []).append(int(idx))

        cylindrical_radius = np.hypot(x, y)

        for key in sorted(groups):
            group_indices = np.asarray(groups[key], dtype=np.int64)
            # Closest layer to the detector middle, then highest energy.
            order = np.lexsort(
                (-energy[group_indices], np.abs(layer[group_indices] - middle_layer))
            )
            representative = int(group_indices[order[0]])
            merged_position = np.array(
                [x[representative], y[representative], z[representative]],
                dtype=np.float64,
            )
            source_cyl_radius = cylindrical_radius[representative]
            if source_cyl_radius > 0.0:
                merged_position *= middle_cyl_radius_mm / source_cyl_radius

            for field in fields:
                if field == "energy":
                    value = np.sum(values[field][group_indices], dtype=np.float64)
                elif field == "layer":
                    value = output_layer
                elif field == "x":
                    value = merged_position[0]
                elif field == "y":
                    value = merged_position[1]
                elif field == "z":
                    value = merged_position[2]
                else:
                    value = values[field][representative]
                event_output[field].append(value)

        for field in fields:
            merged_fields[field].append(
                np.asarray(event_output[field], dtype=values[field].dtype)
            )

    return ak.zip({field: ak.Array(events) for field, events in merged_fields.items()})

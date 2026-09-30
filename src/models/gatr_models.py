"""Cluster-level multi-task GATr model and ROOT streaming dataset."""

import math
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import uproot
from gatr import GATr, MLPConfig, SelfAttentionConfig
from gatr.interface import embed_point, extract_point, extract_point_embedding_reg
from torch.utils.data import IterableDataset, get_worker_info


GLOBAL_FEATURE_NAMES = (
    "log_coordinate_rms_mm",
    "log_pca_sigma1_mm",
    "log_pca_sigma2_mm",
    "log_pca_sigma3_mm",
    "log_transverse_rms_mm",
    "first_layer_energy_fraction",
    "log_first_layer_energy_fraction",
)
FIRST_LAYER_FRACTION_KEY = "cluster_first_layer_energy_fraction"


@torch.no_grad()
def cluster_global_features(
    coordinates, energy_fraction, mask, first_layer_fraction, eps=1.0e-6
):
    """Rotation-invariant absolute-scale features plus front-layer energy share.

    Lengths are in mm and computed in float32 from the kept hits. The
    transverse RMS is taken about the direction from the interaction point
    (origin) to the energy-weighted centroid. The detector has only two
    longitudinal layers, so the longitudinal information is the front-layer
    energy fraction, computed in preprocessing from all hits before
    truncation. Returns a tensor of shape (batch, len(GLOBAL_FEATURE_NAMES)).
    """
    coordinates = coordinates.float()
    w = energy_fraction.float() * mask.float()
    w = w / w.sum(dim=1, keepdim=True).clamp_min(eps)
    mask_f = mask.float()
    n_hits = mask_f.sum(dim=1).clamp_min(1.0)

    unweighted_centre = (coordinates * mask_f.unsqueeze(-1)).sum(1) / n_hits.unsqueeze(-1)
    unweighted = (coordinates - unweighted_centre.unsqueeze(1)) * mask_f.unsqueeze(-1)
    coordinate_rms = (unweighted.square().sum(dim=(1, 2)) / (3.0 * n_hits)).sqrt()

    centroid = (coordinates * w.unsqueeze(-1)).sum(dim=1)
    relative = coordinates - centroid.unsqueeze(1)
    covariance = torch.einsum("bn,bni,bnj->bij", w, relative, relative)
    eigenvalues = torch.linalg.eigvalsh(covariance).flip(-1).clamp_min(0.0)
    pca_sigma = eigenvalues.sqrt()

    axis = centroid / centroid.norm(dim=-1, keepdim=True).clamp_min(eps)
    depth_rel = (relative * axis.unsqueeze(1)).sum(dim=-1)
    transverse_sq = (relative.square().sum(dim=-1) - depth_rel.square()).clamp_min(0.0)
    sigma_t = (w * transverse_sq).sum(dim=1).sqrt()

    first_layer = first_layer_fraction.float().reshape(-1).clamp(0.0, 1.0)
    log_len = lambda value: torch.log(value.clamp_min(1.0e-3))
    return torch.cat(
        (
            log_len(coordinate_rms).unsqueeze(-1),
            log_len(pca_sigma),
            log_len(sigma_t).unsqueeze(-1),
            first_layer.unsqueeze(-1),
            torch.log(first_layer + 1.0e-4).unsqueeze(-1),
        ),
        dim=-1,
    )


class CalorimeterGATrClassifier(nn.Module):
    """Joint cluster classifier, LLP mass regressor, and decay-point regressor.

    Hit coordinates are centred and isotropically scaled independently for each
    cluster. The decay head predicts a point in those local coordinates and then
    applies the inverse transform, yielding an absolute point in millimetres.

    With ``use_global_features`` the scalar heads additionally receive
    batch-normalised cluster-level features (absolute spatial scale, which the
    per-cluster coordinate normalisation otherwise removes, and the front-layer
    energy fraction; see ``cluster_global_features``). ``forward`` then needs
    ``first_layer_fraction``.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = dict(config)
        self.eps = float(config.get("normalization_eps", 1.0e-8))

        in_s_channels = int(config.get("in_s_channels", 2))
        if in_s_channels != 2:
            raise ValueError(
                "The model constructs exactly two scalar input channels; "
                "set 'in_s_channels' to 2."
            )

        self.gatr = GATr(
            in_mv_channels=config.get("in_mv_channels", 1),
            out_mv_channels=config.get("out_mv_channels", 1),
            hidden_mv_channels=config.get("hidden_mv_channels", 8),
            in_s_channels=in_s_channels,
            out_s_channels=config.get("out_s_channels", 16),
            hidden_s_channels=config.get("hidden_s_channels", 32),
            num_blocks=config.get("num_blocks", 2),
            attention=SelfAttentionConfig(
                num_heads=int(config.get("num_heads", 8))
            ),
            mlp=MLPConfig(),
            dropout_prob=config.get("dropout", 0.1),
        )

        if int(config.get("out_mv_channels", 1)) < 1:
            raise ValueError("At least one output multivector channel is required.")

        out_s = int(config.get("out_s_channels", 16))
        hidden = int(config.get("mlp_hidden_dim", 32))
        dropout = float(config.get("dropout", 0.1))
        # Defaults to False so that checkpoints without the key keep loading.
        self.use_global_features = bool(config.get("use_global_features", False))
        n_global = len(GLOBAL_FEATURE_NAMES) if self.use_global_features else 0
        if self.use_global_features:
            self.global_norm = nn.BatchNorm1d(n_global)
        pooled_dim = 2 * out_s + 1 + n_global

        def scalar_head(output_dim):
            return nn.Sequential(
                nn.Linear(pooled_dim, hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, output_dim),
            )

        self.classifier = scalar_head(1)
        # Predict log(1 + mass / GeV); the transformed target has a stable scale
        # across samples spanning a broad LLP mass range.
        self.mass_regressor = scalar_head(1)

    def _normalise_cluster(self, x, y, z, mask):
        """Return local coordinates and the invertible per-cluster transform."""
        coordinates = torch.stack((x, y, z), dim=-1)
        mask_f = mask.unsqueeze(-1).to(coordinates.dtype)
        n_hits = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        centre = (coordinates * mask_f).sum(dim=1, keepdim=True) / n_hits
        centred = (coordinates - centre) * mask_f
        variance = centred.square().sum(dim=(1, 2), keepdim=True) / (3.0 * n_hits)
        scale = variance.clamp_min(self.eps).sqrt()
        return centred / scale, centre, scale

    def forward(self, x, y, z, energy, mask=None, first_layer_fraction=None):
        """Return classification and regression predictions for each cluster."""
        if mask is None:
            mask = energy > 0
        mask = mask.bool()
        if torch.any(mask.sum(dim=1) == 0):
            raise ValueError("Every cluster must contain at least one valid hit.")

        coordinates, centre, coordinate_scale = self._normalise_cluster(
            x, y, z, mask
        )
        mv_in = embed_point(coordinates).unsqueeze(-2)

        mask_f = mask.to(energy.dtype)
        valid_energy = energy.clamp_min(0.0) * mask_f
        total_energy = valid_energy.sum(dim=1, keepdim=True).clamp_min(self.eps)
        energy_fraction = valid_energy / total_energy
        log_total_energy = torch.log1p(total_energy)
        s_in = torch.stack(
            (
                energy_fraction,
                log_total_energy.expand_as(energy_fraction) * mask_f,
            ),
            dim=-1,
        )

        n_tokens = mask.shape[1]
        attention_mask = mask.unsqueeze(1).unsqueeze(1).expand(
            -1, 1, n_tokens, -1
        )
        mv_out, s_out = self.gatr(
            mv_in, s_in, attention_mask=attention_mask
        )

        mean_weights = mask_f / mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled_mean = torch.sum(s_out * mean_weights.unsqueeze(-1), dim=1)
        pooled_energy = torch.sum(
            s_out * energy_fraction.unsqueeze(-1), dim=1
        )
        pooled_parts = [pooled_mean, pooled_energy, log_total_energy]
        if self.use_global_features:
            if first_layer_fraction is None:
                raise ValueError(
                    "use_global_features requires first_layer_fraction "
                    f"(branch '{FIRST_LAYER_FRACTION_KEY}'); re-run preprocessing."
                )
            with torch.autocast(device_type=x.device.type, enabled=False):
                global_features = self.global_norm(
                    cluster_global_features(
                        torch.stack((x, y, z), dim=-1),
                        energy_fraction,
                        mask,
                        first_layer_fraction,
                    )
                )
            pooled_parts.append(global_features.to(pooled_mean.dtype))
        pooled_scalars = torch.cat(pooled_parts, dim=-1)

        classification_logits = self.classifier(pooled_scalars).squeeze(-1)
        mass_log1p = self.mass_regressor(pooled_scalars).squeeze(-1)
        mass_gev = torch.expm1(mass_log1p.clamp(max=20.0)).clamp_min(0.0)

        # Energy-weighted linear pooling preserves the multivector grade
        # structure. The first output channel is interpreted as a local point.
        pooled_mv = torch.sum(
            mv_out * energy_fraction.unsqueeze(-1).unsqueeze(-1), dim=1
        )
        decay_local = extract_point(pooled_mv[:, 0, :])
        decay_point_mm = (
            centre.squeeze(1)
            + decay_local * coordinate_scale.reshape(-1, 1)
        )
        point_embedding_reg = extract_point_embedding_reg(
            pooled_mv[:, 0, :]
        ).squeeze(-1)

        return {
            "classification_logits": classification_logits,
            "mass_log1p": mass_log1p,
            "mass_GeV": mass_gev,
            "decay_point_mm": decay_point_mm,
            "point_embedding_reg": point_embedding_reg,
        }


class ROOTShowerDataset(IterableDataset):
    """Stream batches of preprocessed, fixed-size CLUE cluster rows.

    ``max_hits=None`` uses the padded hit width stored in the files. With
    ``shuffle=True`` the file order and the row order inside each read step
    are permuted with a seed that changes with ``set_epoch``. ``drop_last``
    skips the final incomplete batch (BatchNorm needs more than one sample).
    """

    def __init__(
        self,
        root_file_paths: list[str],
        tree_name: str = "CLUEShowers",
        extra_tree_name: str = "CLUEExtra",
        batch_size: int = 32,
        max_hits: int | None = None,
        read_step_size: int = 1024,
        require_classification_valid: bool = True,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ):
        super().__init__()
        self.file_paths = list(root_file_paths)
        self.tree_name = tree_name
        self.extra_tree_name = extra_tree_name
        self.batch_size = batch_size
        self.read_step_size = read_step_size
        self.require_classification_valid = require_classification_valid
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.max_hits = max_hits or self._infer_max_hits()
        self.target_keys = (
            "event_label",
            "classification_valid",
            "regression_valid",
            "mass_valid",
            "decay_vertex_valid",
            "llp_pdg",
            "llp_mass_GeV",
            "llp_energy_GeV",
            "llp_decay_x_mm",
            "llp_decay_y_mm",
            "llp_decay_z_mm",
            "llp_decay_dx_mm",
            "llp_decay_dy_mm",
            "llp_decay_dz_mm",
            "llp_decay_length_mm",
            "llp_proper_time_ns",
            "cluster_x",
            "cluster_y",
            "cluster_z",
            "cluster_energy",
            FIRST_LAYER_FRACTION_KEY,
        )
        self.total_events = self._count_total_events()

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _infer_max_hits(self):
        for path in self.file_paths:
            with uproot.open(path) as root_file:
                if self.tree_name in root_file:
                    mask = root_file[self.tree_name]["hit_mask"].array(
                        entry_stop=1, library="np"
                    )
                    return int(mask.shape[1])
        raise FileNotFoundError("Cannot infer max_hits: no readable input file.")

    def _count_total_events(self):
        """Count cluster rows that will be yielded (one cluster per row)."""
        total = 0
        for path in self.file_paths:
            try:
                with uproot.open(path) as root_file:
                    if self.tree_name not in root_file:
                        continue
                    tree = root_file[self.tree_name]
                    if (
                        self.require_classification_valid
                        and "classification_valid" in tree.keys()
                    ):
                        total += int(
                            tree["classification_valid"]
                            .array(library="np")
                            .sum()
                        )
                    else:
                        total += tree.num_entries
            except Exception:
                continue
        return total

    def __len__(self):
        if self.drop_last:
            return self.total_events // self.batch_size
        return math.ceil(self.total_events / self.batch_size)

    def _make_cluster_sample(
        self,
        x,
        y,
        z,
        energy,
        layer,
        cluster_ids,
        event_mask,
        cluster_id,
        label,
    ):
        select = event_mask & (cluster_ids == cluster_id) & (energy > 0)
        indices = np.flatnonzero(select)
        if len(indices) > self.max_hits:
            indices = indices[
                np.argsort(energy[indices])[::-1][: self.max_hits]
            ]

        n_hits = len(indices)
        sample = {
            "x": np.zeros(self.max_hits, dtype=np.float32),
            "y": np.zeros(self.max_hits, dtype=np.float32),
            "z": np.zeros(self.max_hits, dtype=np.float32),
            "energy_raw": np.zeros(self.max_hits, dtype=np.float32),
            "layer": np.zeros(self.max_hits, dtype=np.float32),
            "mask": np.zeros(self.max_hits, dtype=np.bool_),
            "label": np.float32(label),
            "cluster_id": np.int64(cluster_id),
        }
        sample["x"][:n_hits] = x[indices]
        sample["y"][:n_hits] = y[indices]
        sample["z"][:n_hits] = z[indices]
        sample["energy_raw"][:n_hits] = energy[indices]
        sample["layer"][:n_hits] = layer[indices]
        sample["mask"][:n_hits] = True
        return sample

    @staticmethod
    def _collate(samples):
        tensors = {
            key: torch.from_numpy(np.stack([sample[key] for sample in samples]))
            for key in ("x", "y", "z", "energy_raw", "layer", "mask")
        }
        tensors["label"] = torch.from_numpy(
            np.asarray(
                [sample["label"] for sample in samples], dtype=np.float32
            )
        )
        tensors["cluster_id"] = torch.from_numpy(
            np.asarray(
                [sample["cluster_id"] for sample in samples], dtype=np.int64
            )
        )
        for key in samples[0]:
            if key in tensors or key in {"cluster_id", "label"}:
                continue
            values = np.asarray([sample[key] for sample in samples])
            tensors[key] = torch.from_numpy(values)
        return tensors

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker_info = get_worker_info()
        rng = np.random.default_rng(self.seed + 1000003 * self.epoch)
        file_paths = list(self.file_paths)
        if self.shuffle:
            rng.shuffle(file_paths)
        if worker_info is None:
            files_to_process = file_paths
        else:
            per_worker = math.ceil(
                len(self.file_paths) / worker_info.num_workers
            )
            start = worker_info.id * per_worker
            files_to_process = file_paths[start : start + per_worker]

        pending = []
        for path in files_to_process:
            with uproot.open(path) as root_file:
                if (
                    self.tree_name not in root_file
                    or self.extra_tree_name not in root_file
                ):
                    raise KeyError(
                        f"{path} must contain both '{self.tree_name}' and "
                        f"'{self.extra_tree_name}' for cluster-level training."
                    )
                main_tree = root_file[self.tree_name]
                extra_tree = root_file[self.extra_tree_name]
                main_branches = [
                    "hit_E_raw",
                    "hit_layer",
                    "hit_mask",
                    "label",
                ]
                main_branches += [
                    key
                    for key in self.target_keys
                    if key in main_tree.keys()
                ]
                main_iter = main_tree.iterate(
                    main_branches,
                    step_size=self.read_step_size,
                    library="np",
                )
                extra_iter = extra_tree.iterate(
                    ["hit_x", "hit_y", "hit_z", "hit_cluster_id"],
                    step_size=self.read_step_size,
                    library="np",
                )

                for main, extra in zip(main_iter, extra_iter):
                    row_order = np.arange(len(main["label"]))
                    if self.shuffle:
                        rng.shuffle(row_order)
                    for event_idx in row_order:
                        label = main["label"][event_idx]
                        if (
                            self.require_classification_valid
                            and "classification_valid" in main
                            and not bool(
                                main["classification_valid"][event_idx]
                            )
                        ):
                            continue
                        energy = main["hit_E_raw"][event_idx]
                        event_mask = main["hit_mask"][event_idx].astype(bool)
                        cluster_ids = extra["hit_cluster_id"][event_idx]
                        valid_ids = np.unique(
                            cluster_ids[
                                event_mask
                                & (energy > 0)
                                & (cluster_ids >= 0)
                            ]
                        )
                        for cluster_id in valid_ids:
                            sample = self._make_cluster_sample(
                                extra["hit_x"][event_idx],
                                extra["hit_y"][event_idx],
                                extra["hit_z"][event_idx],
                                energy,
                                main["hit_layer"][event_idx],
                                cluster_ids,
                                event_mask,
                                int(cluster_id),
                                float(label),
                            )
                            for key in self.target_keys:
                                if key in main:
                                    sample[key] = main[key][event_idx]
                            pending.append(sample)
                            if len(pending) == self.batch_size:
                                yield self._collate(pending)
                                pending = []

        if pending and not self.drop_last:
            yield self._collate(pending)


class GATrWrapper:
    """Expose classification probabilities through predict_proba."""

    def __init__(self, model, device=None, amp_dtype=torch.bfloat16):
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.amp_dtype = amp_dtype
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_proba(self, cluster_dataloader):
        all_probs = []
        use_amp = self.device.type == "cuda" and self.amp_dtype is not None
        for batch in cluster_dataloader:
            inputs = [
                batch[key].to(self.device)
                for key in ("x", "y", "z", "energy_raw", "mask")
            ]
            first_layer = batch.get(FIRST_LAYER_FRACTION_KEY)
            if first_layer is not None:
                first_layer = first_layer.to(self.device)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype or torch.float32,
                enabled=use_amp,
            ):
                outputs = self.model(
                    *inputs[:4], mask=inputs[4],
                    first_layer_fraction=first_layer,
                )
                all_probs.append(
                    torch.sigmoid(
                        outputs["classification_logits"]
                    ).cpu()
                )

        if not all_probs:
            return np.empty((0, 2), dtype=np.float32)
        signal_probs = torch.cat(all_probs, dim=0).numpy()
        return np.column_stack((1.0 - signal_probs, signal_probs))

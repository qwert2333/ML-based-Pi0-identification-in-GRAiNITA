import numpy as np
import math
import uproot
import torch
import torch.nn as nn
from gatr import GATr,MLPConfig, SelfAttentionConfig
from torch.utils.data import IterableDataset, get_worker_info


class CalorimeterGATrClassifier(nn.Module):
    """
    Modular E(3) GATr model for single photon (gamma) vs neutral pion (pi0) 
    shower classification from calorimeter readouts.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
                

        self.gatr = GATr(
            in_mv_channels=config.get("in_mv_channels", 1),
            out_mv_channels=config.get("out_mv_channels", 1),
            hidden_mv_channels=config.get("hidden_mv_channels", 8),
            in_s_channels=config.get("in_s_channels", 1), ##For debugging set to 1 , otherweise 3 for energy, time, layer
            out_s_channels=config.get("out_s_channels", 16),  
            hidden_s_channels=config.get("hidden_s_channels", 32),
            num_blocks=config.get("num_blocks", 2),
            attention=SelfAttentionConfig(),  
            mlp=MLPConfig(), 
        )
        
        # 3. Aggregation & Output Heads
        # Fetch output channel counts (matching GATr's output dimensions)
        out_mv = config.get("out_mv_channels", 1)
        out_s = config.get("out_s_channels", 16)

        # Output multivector dimension = out_mv_channels * 16 multivector components
        mv_dim = out_mv * 16
        s_dim = out_s

        mlp_hidden = config.get("mlp_hidden_dim", 32)
        dropout_p = config.get("dropout", 0.1)

        # Project combined geometric multivector + scalar embeddings to logit score
        self.classifier = nn.Sequential(
            nn.Linear(mv_dim + s_dim, mlp_hidden),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(mlp_hidden, 1),  # Logit output for binary class
        )

    def format_multivectors(self, x, y, z, energy):
        """Converts raw spatial (x,y,z,E) inputs into PGA G_3,0,1 multivectors."""
        batch_size, num_hits = x.shape
        mv = torch.zeros(batch_size, num_hits, 1, 16, device=x.device, dtype=x.dtype)
        
        # Index 0: 0-blade (scalar energy)
        # Normalize hit energies per shower before putting into multivector
        E_sum = energy.sum(dim=-1, keepdim=True) + 1e-8
        energy_norm = energy / E_sum

        mv[..., 0, 0] = energy_norm  # Pass normalized energy fraction
        # Indices 1, 2, 3: 1-blades (x, y, z spatial positions)
        mv[..., 0, 1] = x
        mv[..., 0, 2] = y
        mv[..., 0, 3] = z
        return mv

    def forward(self, x, y, z, energy, extra_scalars=None, mask=None):
        """
        Args:
            x, y, z: (B, N) spatial positions of cell hits
            energy:  (B, N) deposited energy in hit cells
            extra_scalars: (B, N, C_extra) optional extra hit features (timing, layer, etc.)
            mask:    (B, N) boolean mask (True for real hits, False for padded hits)
        """
        # --- Preprocessing & Encoding ---
        mv_in = self.format_multivectors(x, y, z, energy)  # (B, N, 1, 16)
        
        # Construct auxiliary scalar input tensor
        if extra_scalars is None:
            s_in = energy.unsqueeze(-1)  # (B, N, 1)
        else:
            s_in = torch.cat([energy.unsqueeze(-1), extra_scalars], dim=-1)

        # --- GATr Backbone ---
        mv_out, s_out = self.gatr(mv_in, s_in)  # mv_out: (B, N, C_mv, 16), s_out: (B, N, C_s)
        
        # --- Energy-Weighted Pooling ---
        # Weight hit features by relative hit energy so low-energy noise is suppressed
        weights = energy.unsqueeze(-1)  # (B, N, 1)
        if mask is not None:
            weights = weights * mask.unsqueeze(-1)
            
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Flatten multivector channels into vector: (B, N, C_mv * 16)
        B, N, C_mv, _ = mv_out.shape
        mv_flat = mv_out.view(B, N, C_mv * 16)
        
        # Pool across hits dimension (N)
        pooled_mv = torch.sum(mv_flat * weights, dim=1)  # (B, C_mv * 16)
        pooled_s = torch.sum(s_out * weights, dim=1)    # (B, C_s)
        
        # --- Output Classification Head ---
        pooled_features = torch.cat([pooled_mv, pooled_s], dim=-1)
        logits = self.classifier(pooled_features)
        return logits.squeeze(-1)


class ROOTShowerDataset(IterableDataset):
    """Streams fixed-size preprocessed calorimeter showers directly from ROOT files."""
    def __init__(self, root_file_paths: list[str], tree_name: str = "CLUEShowers", batch_size: int = 32):
        super().__init__()
        self.file_paths = list(root_file_paths)
        self.tree_name = tree_name
        self.batch_size = batch_size
        self.total_events = self._count_total_events()

    def _count_total_events(self):
        total = 0
        for path in self.file_paths:
            try:
                with uproot.open(path) as f:
                    if self.tree_name in f:
                        total += f[self.tree_name].num_entries
            except Exception:
                continue
        return total

    def __len__(self):
        """Returns the total number of batches in the dataset."""
        return math.ceil(self.total_events / self.batch_size)
    def __iter__(self):
        """Yields batches of shower data as dictionaries of tensors."""
        worker_info = get_worker_info()
        if worker_info is None:
            # Single process loading
            files_to_process = self.file_paths
        else:
                # Safely split files across DataLoader worker threads
                per_worker = int(math.ceil(len(self.file_paths) / float(worker_info.num_workers)))
                worker_id = worker_info.id
                files_to_process = self.file_paths[
                    worker_id * per_worker : (worker_id + 1) * per_worker
                ]
        for path in files_to_process:
            with uproot.open(path) as f:
                if self.tree_name not in f:
                    continue
                tree = f["CLUEShowers"]
                # Stream batches directly as NumPy dicts
                for batch in tree.iterate(step_size=self.batch_size, library="np"):
                    x = torch.from_numpy(batch["hit_x_norm"])
                    y = torch.from_numpy(batch["hit_y_norm"])
                    z = torch.from_numpy(batch["hit_z_norm"])
                    E_raw = torch.from_numpy(batch["hit_E_raw"])
                    E_norm = torch.from_numpy(batch["hit_E_norm"])
                    
                    # Safely handle hit_time_norm if it doesn't exist in the ROOT file
                    if "hit_time_norm" in batch:
                        t_norm = torch.from_numpy(batch["hit_time_norm"])
                    else:
                        # Create zero tensor matching hit_x_norm shape if timing isn't available
                        t_norm = torch.zeros_like(torch.from_numpy(batch["hit_x_norm"]))
                    mask = torch.from_numpy(batch["hit_mask"])
                    labels = torch.from_numpy(batch["label"]).float()

                    # Combine scalar extra features: [Normalized E, Normalized Time]
                    extra_scalars = torch.stack([E_norm, t_norm], dim=-1)

                    yield {
                        "x": x,
                        "y": y,
                        "z": z,
                        "energy_raw": E_raw,  # For energy-weighted pooling
                        "extra_scalars": extra_scalars,
                        "mask": mask,
                        "label": labels
                    }

class GATrWrapper:
    """
    Adapter wrapper that exposes a `.predict_proba()` method for PyTorch GATr,
    making it 100% compatible with evaluate_bdt().
    """
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_proba(self, test_dataloader):
        """
        Accepts a ROOTShowerDataset DataLoader, runs model inference,
        and returns a numpy array of shape (N, 2) matching sklearn predict_proba output.
        """
        all_probs = []
        for batch in test_dataloader:
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            z = batch["z"].to(self.device)
            energy_raw = batch["energy_raw"].to(self.device)
            extra_scalars = None  #batch["extra_scalars"].to(self.device)
            mask = batch["mask"].to(self.device)

            with torch.cuda.amp.autocast():
                #print(f"DEBUG: x shape: {x.shape}, y shape: {y.shape}, z shape: {z.shape}")
                logits = self.model(x, y, z, energy_raw, extra_scalars=extra_scalars, mask=mask)
                probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu())

        sig_probs = torch.cat(all_probs, dim=0).numpy()
        bkg_probs = 1.0 - sig_probs
        
        
        return np.column_stack([bkg_probs, sig_probs])
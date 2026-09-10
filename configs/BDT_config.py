# Feature variables used for BDT training
TRAIN_VARS = [
    "n_hits",
    "n_clusters",
    "total_reco_energy",
    "avg_hit_energy",
    "mean_layer",
    "sigma_layer",
    "shower_radius",
    "leading_cluster_ratio",
    "mean_rho",
    "mean_delta",
]
EXTENDED_TRAIN_VARS_truncated = [
    "n_hits",
    "n_clusters",
    "total_reco_energy",
    "leading_cluster_energy",
    "leading_cluster_ratio",
    "max_hit_energy_ratio",
    "ratio_e_max_2ndmax",
    "delta_e_2ndmax_min",
    "shower_radius",
    "width_x",
    "width_y",
    "e_fr_side",
    "mean_rho",
    "max_rho",
]

EXTENDED_TRAIN_VARS = [
    "n_hits",
    "n_clusters",
    "total_reco_energy",
    "leading_cluster_energy",
    "leading_cluster_ratio",
    "max_hit_energy_ratio",
    "ratio_e_max_2ndmax",
    "delta_e_2ndmax_min",
    "shower_radius",
    "width_x",
    "width_y",
    "e_fr_side",
    "mean_layer",
    "sigma_layer",
    "mean_rho",
    "max_rho",
    "mean_delta",
    "max_delta",
]
# Label & metadata variables
TARGET_PDG = "target_pdg"
TARGET_ENERGY = "target_energy"
LABEL_COL = "label"  

TREE_NAME = "events"
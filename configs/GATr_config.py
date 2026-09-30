#---------------------------------------------------------------------------------------------------------------------------
# GATr model configuration constants
#---------------------------------------------------------------------------------------------------------------------------

DEBUG_CONFIG = {
    "in_s_channels": 2,           # [Hit energy fraction, log(total cluster energy)]
    "hidden_mv_channels": 4,      # Small 4-channel multivector space
    "hidden_s_channels": 16,     # Small scalar hidden dim
    "num_blocks": 2,              # 2 GATr blocks
    "num_heads": 2,               # 2 attention heads
    "mlp_hidden_dim": 32,
    "dropout": 0.0,
    "out_mv_channels": 1,
    "classification_loss_weight": 1.0,
    "mass_loss_weight": 1.0,
    "decay_point_loss_weight": 1.0,
    "point_embedding_reg_weight": 1.0e-3,
    "decay_point_scale_mm": 1000.0,
}

SCALE_UP_CONFIG = {
    "in_s_channels": 2,
    "hidden_mv_channels": 16,     # High-capacity multivector space
    "hidden_s_channels": 64,      # Deep scalar representation
    "num_blocks": 8,              # 8 GATr blocks
    "num_heads": 8,               # 8 attention heads
    "mlp_hidden_dim": 128,
    "dropout": 0.1,
    "out_mv_channels": 1,
    "classification_loss_weight": 1.0,
    "mass_loss_weight": 1.0,
    "decay_point_loss_weight": 1.0,
    "point_embedding_reg_weight": 1.0e-3,
    "decay_point_scale_mm": 1000.0,
}
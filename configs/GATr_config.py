#---------------------------------------------------------------------------------------------------------------------------
# GATr model configuration constants
#---------------------------------------------------------------------------------------------------------------------------

DEBUG_CONFIG = {
    "in_scalar_channels": 1,      # [Energy, Cell Time]
    "hidden_mv_channels": 4,      # Small 4-channel multivector space
    "hidden_s_channels": 16,     # Small scalar hidden dim
    "num_blocks": 2,              # 2 GATr blocks
    "num_heads": 2,               # 2 attention heads
    "mlp_hidden_dim": 32,
    "dropout": 0.0,
}

SCALE_UP_CONFIG = {
    "in_scalar_channels": 2,
    "hidden_mv_channels": 16,     # High-capacity multivector space
    "hidden_s_channels": 64,      # Deep scalar representation
    "num_blocks": 8,              # 8 GATr blocks
    "num_heads": 8,               # 8 attention heads
    "mlp_hidden_dim": 128,
    "dropout": 0.1,
}
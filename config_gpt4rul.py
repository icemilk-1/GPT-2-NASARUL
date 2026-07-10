"""GPT4RUL — Strict paper reproduction config (Tan et al., QR2MSE 2025).

Paper: "Pre-trained LLM-based remaining useful life prediction of aircraft engines"

Architecture (Section 2.3, Figure 1):
  Sensor window → Patching → PCF blocks × N → Linear(128→768)
  → Frozen GPT-2(3 layers) → Add & LayerNorm → Flatten & LayerNorm → Linear → RUL

Normalization (Section 3.2.1):
  KMeans clustering on settings → per-cluster Z-score → global MinMax [0,1]

Feature selection (Section 2.2):
  Corr(0.5) + Mono(0.5) combined score, keep sensors above mean

Data split: temporal 80/20 by cycle (前 80% 时间 → train, 后 20% → val)
"""

from __future__ import annotations

from pathlib import Path

# =============================================================================
#  Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "CMaps"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
ABLATION_OUTPUT_DIR = OUTPUT_DIR / "ablation"
ABLATION_CHECKPOINT_DIR = ABLATION_OUTPUT_DIR / "checkpoints"
ABLATION_RESULTS_DIR = RESULTS_DIR / "ablation"

ALL_DATASETS = ["FD001", "FD002", "FD003", "FD004"]
COLS = ["id", "cycle", "setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
SETTING_COLS = ["setting1", "setting2", "setting3"]

# =============================================================================
#  Data preprocessing (Section 3.2)
# =============================================================================

# KMeans clustering on settings
# FD001/FD003: single operating condition → n_clusters=1
# FD002/FD004: multiple conditions → n_clusters=6
REGIME_DATASETS = {"FD002", "FD004"}
N_CLUSTERS = 6

# RUL cap
RUL_CAP = 125.0

# Feature selection: Corr+Mono combo score
FEATURE_SELECT_WEIGHT_CORR = 0.5
FEATURE_SELECT_WEIGHT_MONO = 0.5
# Force top-N selection (None = use Cri > mean threshold)
N_SELECTED_FEATURES: dict[str, int | None] = {
    "FD001": None, "FD002": None, "FD003": None, "FD004": 14,
}

# =============================================================================
#  Per-dataset hyperparameters (Table 2)
# =============================================================================

# Window & stride for sliding-window preprocessing
WINDOW_LENGTH: dict[str, int] = {
    "FD001": 30, "FD002": 64, "FD003": 40, "FD004": 64,
}
WINDOW_STRIDE: dict[str, int] = {
    "FD001": 1, "FD002": 1, "FD003": 1, "FD004": 1,   # paper Section 3.2.3: step=1
}

# Patching (Section 2.3.1)
PATCH_SIZE = 8
PATCH_STRIDE: dict[str, int] = {
    "FD001": 4, "FD002": 8, "FD003": 8, "FD004": 8,
}

# PCF Block (Section 2.3.3, Table 2)
PCF_HIDDEN_DIM = 128
PCF_N_BLOCKS: dict[str, int] = {
    "FD001": 2, "FD002": 2, "FD003": 1, "FD004": 2,  # per paper Table 2
}
PCF_MIXING_FACTOR: dict[str, int] = {
    "FD001": 2, "FD002": 2, "FD003": 1, "FD004": 1,
}

# GPT-2 (Section 2.3.2, Table 2)
GPT2_MODEL_NAME = "openai-community/gpt2"
GPT2_HIDDEN_DIM = 768
GPT2_N_LAYERS = 3                # Only use first 3 layers (Table 2)
GPT2_FREEZE = True               # Always frozen (Section 2.3.2)

# =============================================================================
#  Training (Table 2)
# =============================================================================
BATCH_SIZE: dict[str, int] = {
    "FD001": 64, "FD002": 128, "FD003": 64, "FD004": 128,
}
LEARNING_RATE = 0.005            # paper Table 2
WEIGHT_DECAY = 0.01              # L2 regularization
DROPOUT = 0.2                    # per paper
LR_DECAY_EPOCHS = 10
LR_DECAY_FACTOR = 0.1
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 10
EARLY_STOP_METRIC = "val_rmse"     # val_loss | val_rmse
VAL_MODE = "all"                # all | last_window  (use all windows from val engines)
PCF_STYLE = "sequential"         # sequential | parallel
GRAD_CLIP = 0                    # 0 to disable

# =============================================================================
#  General
# =============================================================================
RANDOM_SEED = 42
VAL_RATIO = 0.2                  # 80/20 train/val split


# ---- Helper functions ----
def n_clusters_for(dataset_id: str) -> int:
    return N_CLUSTERS if dataset_id in REGIME_DATASETS else 1


def patch_stride_for(dataset_id: str) -> int:
    return PATCH_STRIDE.get(dataset_id, 8)


def pcf_n_blocks_for(dataset_id: str) -> int:
    return PCF_N_BLOCKS.get(dataset_id, 2)


def pcf_mixing_factor_for(dataset_id: str) -> int:
    return PCF_MIXING_FACTOR.get(dataset_id, 2)


def batch_size_for(dataset_id: str) -> int:
    return BATCH_SIZE.get(dataset_id, 64)


def window_length_for(dataset_id: str) -> int:
    return WINDOW_LENGTH.get(dataset_id, 30)


def window_stride_for(dataset_id: str) -> int:
    return WINDOW_STRIDE.get(dataset_id, 1)


def n_selected_features_for(dataset_id: str) -> int | None:
    return N_SELECTED_FEATURES.get(dataset_id, None)

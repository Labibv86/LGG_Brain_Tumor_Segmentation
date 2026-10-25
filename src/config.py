"""
Central configuration for LGG brain tumor segmentation.
All hyperparameters and paths live here. Change a value once, it propagates everywhere.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(r"E:\LGG_Tumor_Segmentation")
DATA_ROOT    = PROJECT_ROOT / "data" / "lgg_mri_segmentation"
OUTPUT_DIR   = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
IMAGE_SIZE   = 256          # original slices are 256x256; None would mean no resize
MASK_SUFFIX  = "_mask.tif"
SLICE_SUFFIX = ".tif"

# Patient-level split ratios (must sum to 1.0)
TRAIN_RATIO  = 0.80
VAL_RATIO    = 0.10
TEST_RATIO   = 0.10
SPLIT_SEED   = 42

# DataLoader
BATCH_SIZE   = 16
NUM_WORKERS  = 4            # Windows: start with 4; raise if your CPU is not bottlenecked
PIN_MEMORY   = True

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
IN_CHANNELS  = 3            # RGB tif: pre-contrast / FLAIR / post-contrast
OUT_CHANNELS = 1            # binary tumor mask
BASE_FEATURES = 64          # U-Net first block width

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
DEVICE       = "cuda"       # falls back to cpu automatically if CUDA unavailable
EPOCHS       = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-5

# Loss: combined Dice + BCE. BCE_WEIGHT balances the two terms.
BCE_WEIGHT   = 0.5
DICE_WEIGHT  = 0.5

# LR scheduler (ReduceLROnPlateau on val Dice)
LR_PATIENCE  = 5
LR_FACTOR    = 0.5
MIN_LR       = 1e-6

# Reproducibility
SEED         = 42

# ---------------------------------------------------------------------------
# Evaluation / output
# ---------------------------------------------------------------------------
CHECKPOINT_NAME = "best_model.pth"
METRICS_CSV     = "metrics.csv"
PRED_GRID_NAME  = "test_predictions.png"
CURVES_NAME     = "training_curves.png"
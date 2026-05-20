"""Typed constants for all wandb log keys.

**No bare string key may appear in ``wandb_logger.py`` or anywhere else.**
Always reference the constants defined here.
"""

from typing import Final

# ---------------------------------------------------------------------------
# Fold-level metrics
# ---------------------------------------------------------------------------
FOLD_TRAIN_LOSS: Final[str] = "fold/train/loss"
FOLD_VAL_LOSS: Final[str] = "fold/val/loss"
FOLD_VAL_MAE: Final[str] = "fold/val/mae"
FOLD_VAL_RMSE: Final[str] = "fold/val/rmse"
FOLD_VAL_R2: Final[str] = "fold/val/r2"
FOLD_VAL_PEARSON_R: Final[str] = "fold/val/pearson_r"
FOLD_TEST_MAE: Final[str] = "fold/test/mae"
FOLD_TEST_RMSE: Final[str] = "fold/test/rmse"
FOLD_TEST_R2: Final[str] = "fold/test/r2"
FOLD_TEST_PEARSON_R: Final[str] = "fold/test/pearson_r"

# ---------------------------------------------------------------------------
# Aggregated CV metrics
# ---------------------------------------------------------------------------
CV_MEAN_MAE: Final[str] = "cv/mean/mae"
CV_MEAN_RMSE: Final[str] = "cv/mean/rmse"
CV_MEAN_R2: Final[str] = "cv/mean/r2"
CV_MEAN_PEARSON_R: Final[str] = "cv/mean/pearson_r"
CV_STD_MAE: Final[str] = "cv/std/mae"
CV_STD_RMSE: Final[str] = "cv/std/rmse"
CV_STD_R2: Final[str] = "cv/std/r2"
CV_STD_PEARSON_R: Final[str] = "cv/std/pearson_r"

# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------
MODEL_PARAM_COUNT: Final[str] = "model/param_count"
MODEL_TRAINABLE_PARAM_COUNT: Final[str] = "model/trainable_param_count"
MODEL_NAME: Final[str] = "model/name"
MODEL_BACKBONE: Final[str] = "model/backbone"

# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------
DATASET_NAME: Final[str] = "dataset/name"
DATASET_NUM_SUBJECTS: Final[str] = "dataset/num_subjects"
DATASET_NUM_NODES: Final[str] = "dataset/num_nodes"
DATASET_MODALITY: Final[str] = "dataset/modality"
DATASET_ATLAS: Final[str] = "dataset/atlas"

# ---------------------------------------------------------------------------
# Dataset splits
# ---------------------------------------------------------------------------
FOLD_SPLITS_ARTIFACT_TYPE: Final[str] = "dataset_splits"

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
SWEEP_TRIAL_PARAMS: Final[str] = "sweep/trial/params"
SWEEP_TRIAL_OBJECTIVE: Final[str] = "sweep/trial/objective"

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
TRAIN_EPOCH: Final[str] = "train/epoch"
TRAIN_LR: Final[str] = "train/lr"
TRAIN_BEST_EPOCH: Final[str] = "train/best_epoch"

# ---------------------------------------------------------------------------
# Prediction scatter
# ---------------------------------------------------------------------------
PRED_SCATTER_TABLE: Final[str] = "predictions/scatter"
PRED_Y_TRUE: Final[str] = "predictions/y_true"
PRED_Y_PRED: Final[str] = "predictions/y_pred"

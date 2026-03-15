# ── Training Configuration ────────────────────────────────────────────────────
# Shared hyperparameters for all three models (RoBERTa, DeBERTa, ELECTRA)
# This ensures consistent comparisons across models.

import torch

NUM_LABELS = 10

# Seeds for multi-run variance reporting
SEEDS = [42, 123, 456]

# Unified training arguments (shared by all models)
TRAINING_ARGS = {
    "num_train_epochs": 5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 32,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.06,               # Same for all models
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "logging_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss", # Use val loss to avoid overfitting
    "greater_is_better": False,
    "fp16": torch.cuda.is_available(),
    "save_total_limit": 2,
    "report_to": "none",
}

# Per-model configuration
MODEL_CONFIGS = {
    "roberta": {
        "model_name": "roberta-base",
        "max_length": 128,
    },
    "deberta": {
        "model_name": "microsoft/deberta-v3-base",
        "max_length": 128,
    },
    "electra": {
        "model_name": "google/electra-base-discriminator",
        "max_length": 128,
    },
}

# Data paths (relative to src/training/)
DATA_DIR = "../../data/processed"
TRAIN_FILE = f"{DATA_DIR}/train.parquet"
VAL_FILE = f"{DATA_DIR}/val.parquet"
TEST_FILE = f"{DATA_DIR}/test.parquet"
LABEL_MAPPING_FILE = f"{DATA_DIR}/label_mapping.json"

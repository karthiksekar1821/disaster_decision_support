# Kaggle Run Guide — Disaster Decision Support Pipeline

> **Every code cell below imports and calls functions from the existing project files.**
> No code is duplicated or rewritten — all logic lives in `src/`.

---

## Prerequisites

- Create a Kaggle notebook with **GPU T4 x2** accelerator
- Add your dataset: `karthiksekar1821/crisismmd` (contains `train.json`, `val.json`, `test.json`)
- Add your GitHub repo as a Kaggle dataset or clone it (instructions below)

---

## Cell 1 — Clone Repo & Install Dependencies

```python
import subprocess, os

# Clone the repo (skip if you added it as a Kaggle dataset)
if not os.path.exists("/kaggle/working/disaster_decision_support"):
    subprocess.run([
        "git", "clone",
        "https://github.com/YOUR_USERNAME/disaster_decision_support.git",
        "/kaggle/working/disaster_decision_support"
    ], check=True)

# Install only what Kaggle doesn't already have (captum for attributions)
# Do NOT pin numpy — Kaggle's pre-installed version is compatible
subprocess.run(["pip", "install", "-q", "captum"], check=True)

print("✅ Setup complete.")
```

---

## Cell 2 — Configure Paths for Kaggle

The project files use paths from `config.py`. We override them for Kaggle's directory layout.

```python
import sys, os

# Add source directories to Python path
REPO_DIR = "/kaggle/working/disaster_decision_support"
sys.path.insert(0, os.path.join(REPO_DIR, "src", "training"))
sys.path.insert(0, os.path.join(REPO_DIR, "src", "analysis"))
sys.path.insert(0, os.path.join(REPO_DIR, "src", "evaluation"))
sys.path.insert(0, os.path.join(REPO_DIR, "src", "data"))

# Output directory for models and results
OUTPUT_DIR = "/kaggle/working/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Copy the raw CrisisMMD JSON dataset into the expected location
import shutil
RAW_DIR = os.path.join(REPO_DIR, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)
KAGGLE_DS_DIR = "/kaggle/input/datasets/karthiksekar1821/crisismmd"

for split in ["train.json", "val.json", "test.json"]:
    try:
        shutil.copy(os.path.join(KAGGLE_DS_DIR, split), os.path.join(RAW_DIR, split))
    except FileNotFoundError:
        print(f"Warning: {split} not found in {KAGGLE_DS_DIR}")

PROCESSED_DIR = os.path.join(REPO_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Run data preparation to create parquets and label mappings
print("Running data preparation...")
import subprocess
subprocess.run(["python", "prepare_data.py"], cwd=os.path.join(REPO_DIR, "src", "data"), check=True)

# Override config paths to point to the new processed data
import config

config.DATA_DIR = PROCESSED_DIR
config.TRAIN_FILE = os.path.join(config.DATA_DIR, "train.parquet")
config.VAL_FILE   = os.path.join(config.DATA_DIR, "val.parquet")
config.TEST_FILE  = os.path.join(config.DATA_DIR, "test.parquet")
config.LABEL_MAPPING_FILE = os.path.join(config.DATA_DIR, "label_mapping.json")

print(f"Output dir: {OUTPUT_DIR}")
print(f"NUM_LABELS: {config.NUM_LABELS}")
print(f"SEEDS:      {config.SEEDS}")
```

---

## Cell 3 — Train All Three Models (Multi-Seed)

This uses `train_multi_seed()` from `train_model.py`. Each model is trained 3 times (seeds 42, 123, 456).

> **⏱ Estimated time:** ~2–3 hours per model on T4 GPU. Train one at a time if needed.

```python
from train_model import train_multi_seed

# Train RoBERTa (3 seeds)
roberta_results = train_multi_seed("roberta", OUTPUT_DIR)
```

```python
# Train DeBERTa (3 seeds)
deberta_results = train_multi_seed("deberta", OUTPUT_DIR)
```

```python
# Train ELECTRA (3 seeds)
electra_results = train_multi_seed("electra", OUTPUT_DIR)
```

> **Alternatively**, to train a single seed for quick testing:
> ```python
> from train_model import train_single_model
> result = train_single_model("roberta", seed=42, output_dir=OUTPUT_DIR)
> ```

---

## Cell 4 — Load Saved Predictions

After training, load the saved predictions for the next steps. Uses seed 42 (the primary seed).

```python
import numpy as np
import json

SEED = 42

def load_predictions(model_key, seed, split="test"):
    pred_path = f"{OUTPUT_DIR}/{model_key}/seed_{seed}/predictions/{split}_predictions.npz"
    data = np.load(pred_path)
    return {
        "preds": data["preds"],
        "labels": data["labels"],
        "probs": data["probs"],
    }

# Load test predictions from all 3 models
roberta_test  = load_predictions("roberta",  SEED)
deberta_test  = load_predictions("deberta",  SEED)
electra_test  = load_predictions("electra",  SEED)

# Load validation predictions (needed for meta-learner training)
roberta_val   = load_predictions("roberta",  SEED, split="val")
deberta_val   = load_predictions("deberta",  SEED, split="val")
electra_val   = load_predictions("electra",  SEED, split="val")

# Load label mapping and class names
from train_model import load_label_mapping
label2id, id2label = load_label_mapping()
class_names = [id2label[i] for i in range(len(id2label))]

print(f"Classes: {class_names}")
print(f"Test set size: {len(roberta_test['labels'])}")
```

---

## Cell 5 — Load Tweet Texts (Needed for Ensemble & Attribution)

```python
from datasets import load_dataset

dataset = load_dataset("parquet", data_files={
    "train":      config.TRAIN_FILE,
    "validation": config.VAL_FILE,
    "test":       config.TEST_FILE,
})

val_texts  = dataset["validation"]["tweet_text"]
test_texts = dataset["test"]["tweet_text"]
train_texts  = dataset["train"]["tweet_text"]
train_labels = np.array(dataset["train"]["label"])
test_labels  = np.array(dataset["test"]["label"])

print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
```

---

## Cell 6 — Novelty 1: Dynamic Context-Conditioned Ensemble

Uses `dynamic_ensemble.py` — trains a meta-learner on validation data, evaluates on test data.

```python
from dynamic_ensemble import (
    train_meta_learner,
    predict_with_dynamic_ensemble,
    evaluate_dynamic_ensemble,
    save_ensemble_artifacts,
)

# Probabilities from all 3 models
val_probs_list  = [roberta_val["probs"],  deberta_val["probs"],  electra_val["probs"]]
test_probs_list = [roberta_test["probs"], deberta_test["probs"], electra_test["probs"]]

# Train meta-learner on VALIDATION set
print("Training meta-learner on validation set...")
meta_learner, scaler, feature_names = train_meta_learner(
    val_model_probs_list=val_probs_list,
    val_labels=roberta_val["labels"],  # same labels for all models
    val_tweet_texts=val_texts,
    meta_learner_type="mlp",
)

# Predict on TEST set
print("\nPredicting on test set...")
ensemble_probs, ensemble_preds = predict_with_dynamic_ensemble(
    meta_learner, scaler, test_probs_list, test_texts,
)

# Evaluate
ensemble_metrics = evaluate_dynamic_ensemble(
    ensemble_preds, ensemble_probs, test_labels, class_names,
)

# Save artifacts
ENSEMBLE_DIR = os.path.join(OUTPUT_DIR, "ensemble")
save_ensemble_artifacts(
    meta_learner, scaler, ensemble_probs, ensemble_preds,
    test_labels, ENSEMBLE_DIR,
)
```

---

## Cell 7 — Novelty 2: Class-Adaptive Confidence Thresholds

Uses `adaptive_confidence.py` — sweeps per-class thresholds on validation, applies to test.

```python
from adaptive_confidence import (
    sweep_per_class_thresholds,
    evaluate_selective_prediction,
)

# Sweep thresholds on VALIDATION set using ensemble predictions on val
val_ensemble_probs, val_ensemble_preds = predict_with_dynamic_ensemble(
    meta_learner, scaler, val_probs_list, val_texts,
)

print("Sweeping per-class thresholds on validation set...")
per_class_thresholds, per_class_stats = sweep_per_class_thresholds(
    probs=val_ensemble_probs,
    preds=val_ensemble_preds,
    labels=roberta_val["labels"],
    num_classes=config.NUM_LABELS,
)

# Apply to TEST set
print("\nApplying per-class thresholds to test set...")
selective_results, confidence_accepted_mask = evaluate_selective_prediction(
    probs=ensemble_probs,
    preds=ensemble_preds,
    labels=test_labels,
    per_class_thresholds=per_class_thresholds,
    class_names=class_names,
)

# Save the optimal global threshold (for the Gradio app)
global_threshold = np.mean(list(per_class_thresholds.values()))
CONFIDENCE_DIR = os.path.join(OUTPUT_DIR, "confidence_results")
os.makedirs(CONFIDENCE_DIR, exist_ok=True)
np.save(os.path.join(CONFIDENCE_DIR, "optimal_threshold.npy"), global_threshold)
print(f"\nGlobal threshold (mean of per-class): {global_threshold:.4f}")
```

---

## Cell 8 — Novelty 3: Attribution-Based Reliability Filter

Uses `attribution_filter.py` — flags predictions where top-attributed tokens are irrelevant.

> **⏱ This is slow** (~1–2 sec per sample). Run on a subset first to verify.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from attribution_filter import (
    compute_attributions_for_batch,
    flag_unreliable_predictions,
    combined_abstention,
)

# Load the best RoBERTa model for attribution computation
MODEL_PATH = f"{OUTPUT_DIR}/roberta/seed_{SEED}/best_model"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

# Compute attributions on test set (use a subset for speed)
N_SAMPLES = min(500, len(test_texts))  # Increase for full evaluation
print(f"Computing attributions for {N_SAMPLES} samples...")

attribution_results = compute_attributions_for_batch(
    model=model,
    tokenizer=tokenizer,
    texts=test_texts[:N_SAMPLES],
    predicted_classes=ensemble_preds[:N_SAMPLES],
    device=device,
    n_steps=50,
)

# Flag unreliable predictions
reliability_mask_subset, reliability_details = flag_unreliable_predictions(
    attribution_results=attribution_results,
    probs=ensemble_probs[:N_SAMPLES],
    preds=ensemble_preds[:N_SAMPLES],
    class_names=class_names,
    per_class_thresholds=per_class_thresholds,
)

# Extend mask to full test set (mark un-analyzed samples as reliable)
reliability_mask = np.ones(len(test_labels), dtype=bool)
reliability_mask[:N_SAMPLES] = reliability_mask_subset

# Combined abstention (Novelty 2 + Novelty 3)
combined_mask, combined_results = combined_abstention(
    confidence_accepted_mask=confidence_accepted_mask,
    reliability_mask=reliability_mask,
    preds=ensemble_preds,
    labels=test_labels,
    class_names=class_names,
)
```

---

## Cell 9 — Full Evaluation Report

Uses `evaluation.py` — runs all baselines, McNemar's tests, confusion matrices, and comparisons.

```python
from evaluation import run_full_evaluation

# Prepare model results dict
model_test_results = {
    "RoBERTa": roberta_test,
    "DeBERTa": deberta_test,
    "ELECTRA": electra_test,
}

ensemble_test_results = {
    "preds":  ensemble_preds,
    "labels": test_labels,
    "probs":  ensemble_probs,
}

EVAL_DIR = os.path.join(OUTPUT_DIR, "evaluation_results")

run_full_evaluation(
    model_test_results=model_test_results,
    ensemble_test_results=ensemble_test_results,
    selective_results=selective_results,
    combined_results=combined_results,
    train_texts=train_texts,
    train_labels=train_labels,
    test_texts=test_texts,
    test_labels=test_labels,
    class_names=class_names,
    output_dir=EVAL_DIR,
)
```

---

## Cell 10 — Save Final Models to Kaggle Output

```python
import shutil

# Copy the best models (seed 42) to a clean output location
FINAL_DIR = os.path.join(OUTPUT_DIR, "final_models")
for model_key in ["roberta", "deberta", "electra"]:
    src = f"{OUTPUT_DIR}/{model_key}/seed_{SEED}/best_model"
    dst = f"{FINAL_DIR}/{model_key}/best_model"
    if os.path.exists(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"✅ Saved {model_key} → {dst}")

print("\n✅ All done! Download results from /kaggle/working/output/")
```

---

## File Reference

| Step | Project File Used | Location |
|------|------------------|----------|
| Training | `train_model.py`, `config.py` | `src/training/` |
| Ensemble | `dynamic_ensemble.py`, `context_features.py` | `src/analysis/` |
| Confidence | `adaptive_confidence.py` | `src/analysis/` |
| Attribution | `attribution_filter.py`, `disaster_vocab.py` | `src/analysis/` |
| Evaluation | `evaluation.py` | `src/evaluation/` |

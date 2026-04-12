# Kaggle Run Guide — Disaster Decision Support Pipeline

> **Every code cell below imports and calls functions from the existing project files.**
> No code is duplicated or rewritten — all logic lives in `src/`.

---

## Prerequisites

- Create a Kaggle notebook with **GPU T4 x2** accelerator
- Add your dataset: `karthiksekar1821/humaid` (contains `train.parquet`, `val.parquet`, `test.parquet`)
- Add saved output as input: mount the previous notebook output at `/kaggle/input/main-project/`
- Clone the GitHub repo

---

## Cell 1 — Clone Repo & Install Dependencies

```python
import subprocess, os

if not os.path.exists("/kaggle/working/disaster_decision_support"):
    subprocess.run([
        "git", "clone",
        "https://github.com/YOUR_USERNAME/disaster_decision_support.git",
        "/kaggle/working/disaster_decision_support"
    ], check=True)

print("✅ Setup complete.")
```

---

## Cell 2 — Configure Paths for Kaggle

```python
import os, sys

# Paths
OUTPUT_DIR  = "/kaggle/input/notebooks/karthiksekar1821/main-project/output"
RESULTS_DIR = "/kaggle/working/results"
DATA_DIR    = "/kaggle/input/datasets/karthiksekar1821/humaid"

os.makedirs(RESULTS_DIR, exist_ok=True)

# Add project to path
sys.path.insert(0, "/kaggle/working/disaster_decision_support/src/training")
sys.path.insert(0, "/kaggle/working/disaster_decision_support/src/analysis")
sys.path.insert(0, "/kaggle/working/disaster_decision_support/src/evaluation")
sys.path.insert(0, "/kaggle/working/disaster_decision_support/src/data")
sys.path.insert(0, "/kaggle/working/disaster_decision_support/src/app")

print("OUTPUT_DIR:", OUTPUT_DIR)
print("RESULTS_DIR:", RESULTS_DIR)
print("DATA_DIR:", DATA_DIR)
```

---

## Cell 3 — Train All Six Transformer Models (seed=42)

```python
from train_model import train_single_model

for model_key in ["roberta", "deberta", "electra", "bert", "bertweet", "xtremedistil"]:
    result = train_single_model(model_key, seed=42, output_dir=OUTPUT_DIR)
    print(f"✅ {model_key} done — Test F1: {result['test_metrics']['eval_macro_f1']:.4f}")
```

---

## Cell 4 — Train Non-Transformer Models

```python
import numpy as np
from datasets import load_dataset

# Install captum here (after transformer training) to avoid numpy
# binary incompatibility that would require a kernel restart.
subprocess.run(["pip", "install", "-q", "captum"], check=True)

dataset = load_dataset("parquet", data_files={
    "train":      config.TRAIN_FILE,
    "validation": config.VAL_FILE,
    "test":       config.TEST_FILE,
})

train_texts  = dataset["train"]["tweet_text"]
val_texts    = dataset["validation"]["tweet_text"]
test_texts   = dataset["test"]["tweet_text"]
train_labels = np.array(dataset["train"]["label"])
val_labels   = np.array(dataset["validation"]["label"])
test_labels  = np.array(dataset["test"]["label"])

import torch
weights_path = os.path.join(config.DATA_DIR, "class_weights.pt")
class_weights = torch.load(weights_path).tolist()

# Train CNN
from cnn_classifier import train_cnn
cnn_model, cnn_vocab, cnn_results = train_cnn(
    train_texts, train_labels, val_texts, val_labels,
    test_texts, test_labels, OUTPUT_DIR, class_weights,
)

# Train BiLSTM
from bilstm_classifier import train_bilstm
bilstm_model, bilstm_vocab, bilstm_results = train_bilstm(
    train_texts, train_labels, val_texts, val_labels,
    test_texts, test_labels, OUTPUT_DIR, class_weights,
)
```

---

## Cell 5 — Load All Predictions

```python
import numpy as np, json, pandas as pd

def load_predictions(model_key, split="test"):
    path_with_seed    = f"{OUTPUT_DIR}/{model_key}/seed_42/predictions/{split}_predictions.npz"
    path_without_seed = f"{OUTPUT_DIR}/{model_key}/predictions/{split}_predictions.npz"

    if os.path.exists(path_with_seed):
        data = np.load(path_with_seed)
    elif os.path.exists(path_without_seed):
        data = np.load(path_without_seed)
    else:
        raise FileNotFoundError(
            f"No predictions found for {model_key} ({split})\n"
            f"  Checked: {path_with_seed}\n"
            f"  Checked: {path_without_seed}"
        )
    return {"preds": data["preds"], "labels": data["labels"], "probs": data["probs"]}

# Load predictions from all 8 models
model_keys = ["roberta", "deberta", "electra", "bert", "bertweet", "xtremedistil", "cnn", "bilstm"]
val_preds = {k: load_predictions(k, "val") for k in model_keys}
test_preds = {k: load_predictions(k, "test") for k in model_keys}

# Load tweet texts
val_df   = pd.read_parquet(f"{DATA_DIR}/val.parquet")
test_df  = pd.read_parquet(f"{DATA_DIR}/test.parquet")

val_texts  = val_df["tweet_text"].tolist()[:len(val_preds["roberta"]["preds"])]
test_texts = test_df["tweet_text"].tolist()[:len(test_preds["roberta"]["preds"])]

# Extract labels and class names
test_labels = test_preds["roberta"]["labels"]
val_labels  = val_preds["roberta"]["labels"]
class_names = [
    "infrastructure_and_utility_damage",
    "injured_or_dead_people",
    "not_humanitarian",
    "other_relevant_information",
    "rescue_volunteering_or_donation_effort",
]

print(f"val_texts loaded: {len(val_texts)}")
print(f"test_texts loaded: {len(test_texts)}")
print(f"test_labels shape: {test_labels.shape}")
print(f"class_names: {class_names}")
```

---

## Cell 6 — Model Characterisation (Tweet Styles)

```python
from model_characterisation import (
    classify_tweets_batch, compute_style_performance_matrix,
    plot_style_performance_heatmap,
)

# Classify tweet styles
val_style_labels, _ = classify_tweets_batch(val_texts)
test_style_labels, _ = classify_tweets_batch(test_texts)

# Compute style performance matrix (for transformers)
display_names = {
    "roberta": "RoBERTa", "deberta": "DeBERTa", "electra": "ELECTRA",
    "bert": "BERT", "bertweet": "BERTweet", "xtremedistil": "XtremeDistil",
}
perf_matrix = compute_style_performance_matrix(
    {display_names[k]: val_preds[k] for k in display_names},
    val_style_labels,
    list(display_names.values()),
    save_dir=os.path.join(RESULTS_DIR, "model_style"),
)

EVAL_DIR = os.path.join(RESULTS_DIR, "evaluation_results")
os.makedirs(EVAL_DIR, exist_ok=True)
plot_style_performance_heatmap(
    perf_matrix, list(display_names.values()),
    save_path=os.path.join(EVAL_DIR, "model_style_heatmap.png"),
)
```

---

## Cell 7 — Dynamic Ensemble (8 Models + Style Features)

```python
from dynamic_ensemble import (
    train_meta_learner, predict_with_dynamic_ensemble,
    evaluate_dynamic_ensemble, save_ensemble_artifacts,
)

# Build probability lists (ordered)
val_probs_list  = [val_preds[k]["probs"]  for k in model_keys]
test_probs_list = [test_preds[k]["probs"] for k in model_keys]

# Train meta-learner with style features
meta_learner, scaler, feature_names = train_meta_learner(
    val_model_probs_list=val_probs_list,
    val_labels=val_preds["roberta"]["labels"],
    val_tweet_texts=val_texts,
    val_style_labels=val_style_labels,
    meta_learner_type="mlp",
)

# Predict on test set
ensemble_probs, ensemble_preds = predict_with_dynamic_ensemble(
    meta_learner, scaler, test_probs_list, test_texts,
    test_style_labels=test_style_labels,
)

ensemble_metrics = evaluate_dynamic_ensemble(
    ensemble_preds, ensemble_probs, test_labels, class_names,
)

ENSEMBLE_DIR = os.path.join(RESULTS_DIR, "ensemble")
save_ensemble_artifacts(
    meta_learner, scaler, ensemble_probs, ensemble_preds,
    test_labels, ENSEMBLE_DIR,
)
```

---

## Cell 8 — Novelty 2: Class-Adaptive Confidence Thresholds

```python
from adaptive_confidence import sweep_per_class_thresholds, apply_per_class_thresholds

val_ensemble_probs, val_ensemble_preds = predict_with_dynamic_ensemble(
    meta_learner, scaler, val_probs_list, val_texts,
    test_style_labels=val_style_labels,
)

# Sweep thresholds on validation set
per_class_thresholds = sweep_per_class_thresholds(
    ensemble_probs=val_ensemble_probs,
    ensemble_preds=val_ensemble_preds,
    true_labels=val_labels,
    class_names=class_names,
)

print("\nPer-class thresholds:")
for cls, t in per_class_thresholds.items():
    print(f"  {cls}: {t:.2f}")

# Apply thresholds to test set
selective_results = apply_per_class_thresholds(
    ensemble_probs=ensemble_probs,
    ensemble_preds=ensemble_preds,
    true_labels=test_labels,
    per_class_thresholds=per_class_thresholds,
    class_names=class_names,
)

confidence_accepted_mask = selective_results["accepted_mask"]

# Report
from sklearn.metrics import f1_score as sk_f1
full_f1 = sk_f1(test_labels, ensemble_preds, average="macro")
print(f"\nWithout selective prediction:  Macro F1 = {full_f1:.4f}")
print(f"With class-adaptive thresholds:")
print(f"  Coverage:  {selective_results['coverage']:.4f} ({selective_results['accepted_count']}/{selective_results['total']})")
print(f"  Macro F1:  {selective_results['macro_f1']:.4f}")
print(f"  Accuracy:  {selective_results['accuracy']:.4f}")
print(f"  F1 improvement: +{selective_results['macro_f1'] - full_f1:.4f}")
```

---

## Cell 9 — Novelty 3: Attribution-Based Reliability Filter

```python
import subprocess
subprocess.run(["pip", "install", "-q", "captum"], check=True)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from attribution_filter import compute_attributions_for_batch, flag_unreliable_predictions, combined_abstention

# Load RoBERTa from saved output (try seed_42 path first)
roberta_path = f"{OUTPUT_DIR}/roberta/seed_42/best_model"
if not os.path.exists(roberta_path):
    roberta_path = f"{OUTPUT_DIR}/roberta/best_model"

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(roberta_path)
model = AutoModelForSequenceClassification.from_pretrained(roberta_path)
model.to(device).eval()

N_SAMPLES = min(500, len(test_texts))
attribution_results = compute_attributions_for_batch(
    model=model, tokenizer=tokenizer,
    texts=test_texts[:N_SAMPLES],
    predicted_classes=ensemble_preds[:N_SAMPLES],
    device=device, n_steps=50,
)

reliability_mask_subset, reliability_details = flag_unreliable_predictions(
    attribution_results=attribution_results,
    probs=ensemble_probs[:N_SAMPLES], preds=ensemble_preds[:N_SAMPLES],
    class_names=class_names, per_class_thresholds=per_class_thresholds,
)

reliability_mask = np.ones(len(test_labels), dtype=bool)
reliability_mask[:N_SAMPLES] = reliability_mask_subset

combined_mask, combined_results = combined_abstention(
    confidence_accepted_mask=confidence_accepted_mask,
    reliability_mask=reliability_mask,
    preds=ensemble_preds, labels=test_labels, class_names=class_names,
)
```

---

## Cell 10 — Full Evaluation Report

```python
from evaluation import run_full_evaluation

display_names_all = {
    "roberta": "RoBERTa", "deberta": "DeBERTa", "electra": "ELECTRA",
    "bert": "BERT", "bertweet": "BERTweet", "xtremedistil": "XtremeDistil",
    "cnn": "CNN", "bilstm": "BiLSTM",
}
model_test_results = {display_names_all[k]: test_preds[k] for k in model_keys}
ensemble_test_results = {"preds": ensemble_preds, "labels": test_labels, "probs": ensemble_probs}

run_full_evaluation(
    model_test_results=model_test_results,
    ensemble_test_results=ensemble_test_results,
    selective_results=selective_results,
    combined_results=combined_results,
    train_texts=train_texts, train_labels=train_labels,
    test_texts=test_texts, test_labels=test_labels,
    class_names=class_names,
    output_dir=EVAL_DIR,
    model_output_dir=OUTPUT_DIR,
    style_performance_path=os.path.join(RESULTS_DIR, "model_style", "model_style_performance.json"),
```

---

## Cell 11 — Save Final Models

```python
import shutil

FINAL_DIR = os.path.join(RESULTS_DIR, "final_models")
for model_key in model_keys:
    # Try seed_42 path first
    src = f"{OUTPUT_DIR}/{model_key}/seed_42/best_model"
    if not os.path.exists(src):
        src = f"{OUTPUT_DIR}/{model_key}/best_model"
    dst = f"{FINAL_DIR}/{model_key}/best_model"
    if os.path.exists(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"✅ Saved {model_key} → {dst}")

print(f"\n✅ All done! Results saved to {RESULTS_DIR}")
```

---

## File Reference

| Step | Project File Used | Location |
|------|------------------|----------|
| Data Prep | `prepare_data.py`, `split.py`, `label_utils.py`, `preprocessing.py` | `src/data/` |
| Training | `train_model.py`, `config.py` | `src/training/` |
| CNN | `cnn_classifier.py` | `src/analysis/` |
| BiLSTM | `bilstm_classifier.py` | `src/analysis/` |
| Characterisation | `model_characterisation.py`, `disaster_vocab.py` | `src/analysis/` |
| Ensemble | `dynamic_ensemble.py`, `context_features.py` | `src/analysis/` |
| Confidence | `adaptive_confidence.py` | `src/analysis/` |
| Attribution | `attribution_filter.py`, `disaster_vocab.py` | `src/analysis/` |
| Evaluation | `evaluation.py` | `src/evaluation/` |
| Dashboard | `crisis_dashboard.py` | `src/app/` |

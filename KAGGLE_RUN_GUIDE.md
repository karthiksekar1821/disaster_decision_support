# Kaggle Run Guide — Disaster Decision Support Pipeline

> **Every code cell below imports and calls functions from the existing project files.**
> No code is duplicated or rewritten — all logic lives in `src/`.

---

## Prerequisites

- Create a Kaggle notebook with **GPU T4 x2** accelerator
- Add your dataset: `karthiksekar1821/humaid` (contains `train.parquet`, `val.parquet`, `test.parquet`)
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

# Install dependencies not pre-installed on Kaggle
subprocess.run(["pip", "install", "-q", "captum", "optuna"], check=True)

print("✅ Setup complete.")
```

---

## Cell 2 — Configure Paths for Kaggle

```python
import sys, os

REPO_DIR = "/kaggle/working/disaster_decision_support"
sys.path.insert(0, os.path.join(REPO_DIR, "src", "training"))
sys.path.insert(0, os.path.join(REPO_DIR, "src", "analysis"))
sys.path.insert(0, os.path.join(REPO_DIR, "src", "evaluation"))
sys.path.insert(0, os.path.join(REPO_DIR, "src", "data"))
sys.path.insert(0, os.path.join(REPO_DIR, "src", "app"))

OUTPUT_DIR = "/kaggle/working/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Copy raw data
import shutil
RAW_DIR = os.path.join(REPO_DIR, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)
KAGGLE_DS_DIR = "/kaggle/input/datasets/karthiksekar1821/humaid"

for split in ["train.parquet", "val.parquet", "test.parquet"]:
    try:
        shutil.copy(os.path.join(KAGGLE_DS_DIR, split), os.path.join(RAW_DIR, split))
    except FileNotFoundError:
        print(f"Warning: {split} not found in {KAGGLE_DS_DIR}")

PROCESSED_DIR = os.path.join(REPO_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Run data preparation
print("Running data preparation...")
subprocess.run(["python", "prepare_data.py"], cwd=os.path.join(REPO_DIR, "src", "data"), check=True)

# Override config paths
import config
config.DATA_DIR = PROCESSED_DIR
config.TRAIN_FILE = os.path.join(config.DATA_DIR, "train.parquet")
config.VAL_FILE   = os.path.join(config.DATA_DIR, "val.parquet")
config.TEST_FILE  = os.path.join(config.DATA_DIR, "test.parquet")
config.LABEL_MAPPING_FILE = os.path.join(config.DATA_DIR, "label_mapping.json")

print(f"Output dir: {OUTPUT_DIR}")
print(f"SEED: {config.SEED}")
```

---

## Cell 3 — (Optional) Hyperparameter Tuning

Run Optuna-based tuning per model. Each model runs 20 trials on 30% of data.

> **⏱ ~30-60 min per model on T4 GPU.**

```python
from hyperparameter_tuning import run_hyperparameter_tuning

# Tune one model at a time (optional — skip for default hyperparameters)
# run_hyperparameter_tuning("roberta", OUTPUT_DIR, n_trials=20)
# run_hyperparameter_tuning("deberta", OUTPUT_DIR, n_trials=20)
# ... etc for each model
```

---

## Cell 4 — Train All Six Transformer Models (seed=42)

```python
from train_model import train_single_model

# Train each model — uses tuned hyperparams if available
for model_key in ["roberta", "deberta", "electra", "bert", "bertweet", "xtremedistil"]:
    result = train_single_model(model_key, seed=42, output_dir=OUTPUT_DIR)
    print(f"✅ {model_key} done — Test F1: {result['test_metrics']['eval_macro_f1']:.4f}")
```

---

## Cell 5 — Train Non-Transformer Models

```python
import numpy as np
from datasets import load_dataset

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

## Cell 6 — Load All Predictions

```python
import numpy as np, json

def load_predictions(model_key, split="test"):
    pred_path = f"{OUTPUT_DIR}/{model_key}/predictions/{split}_predictions.npz"
    data = np.load(pred_path)
    return {"preds": data["preds"], "labels": data["labels"], "probs": data["probs"]}

# Load predictions from all 8 models
model_keys = ["roberta", "deberta", "electra", "bert", "bertweet", "xtremedistil", "cnn", "bilstm"]
val_preds = {k: load_predictions(k, "val") for k in model_keys}
test_preds = {k: load_predictions(k, "test") for k in model_keys}

from train_model import load_label_mapping
label2id, id2label = load_label_mapping()
class_names = [id2label[i] for i in range(len(id2label))]

print(f"Classes: {class_names}")
print(f"Test set size: {len(test_preds['roberta']['labels'])}")
```

---

## Cell 7 — Model Characterisation (Tweet Styles)

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
    save_dir=config.DATA_DIR,
)

EVAL_DIR = os.path.join(OUTPUT_DIR, "evaluation_results")
os.makedirs(EVAL_DIR, exist_ok=True)
plot_style_performance_heatmap(
    perf_matrix, list(display_names.values()),
    save_path=os.path.join(EVAL_DIR, "model_style_heatmap.png"),
)
```

---

## Cell 8 — Dynamic Ensemble (8 Models + Style Features)

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

ENSEMBLE_DIR = os.path.join(OUTPUT_DIR, "ensemble")
save_ensemble_artifacts(
    meta_learner, scaler, ensemble_probs, ensemble_preds,
    test_labels, ENSEMBLE_DIR,
)
```

---

## Cell 9 — Novelty 2: Class-Adaptive Confidence Thresholds

```python
from adaptive_confidence import sweep_per_class_thresholds, evaluate_selective_prediction

val_ensemble_probs, val_ensemble_preds = predict_with_dynamic_ensemble(
    meta_learner, scaler, val_probs_list, val_texts,
    test_style_labels=val_style_labels,
)

per_class_thresholds, per_class_stats = sweep_per_class_thresholds(
    probs=val_ensemble_probs, preds=val_ensemble_preds,
    labels=val_preds["roberta"]["labels"], num_classes=config.NUM_LABELS,
)

selective_results, confidence_accepted_mask = evaluate_selective_prediction(
    probs=ensemble_probs, preds=ensemble_preds, labels=test_labels,
    per_class_thresholds=per_class_thresholds, class_names=class_names,
)
```

---

## Cell 10 — Novelty 3: Attribution-Based Reliability Filter

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from attribution_filter import compute_attributions_for_batch, flag_unreliable_predictions, combined_abstention

MODEL_PATH = f"{OUTPUT_DIR}/roberta/best_model"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
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

## Cell 11 — Full Evaluation Report

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
    style_performance_path=os.path.join(config.DATA_DIR, "model_style_performance.json"),
)
```

---

## Cell 12 — Save Final Models

```python
import shutil

FINAL_DIR = os.path.join(OUTPUT_DIR, "final_models")
for model_key in model_keys:
    src = f"{OUTPUT_DIR}/{model_key}/best_model"
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
| Data Prep | `prepare_data.py`, `split.py`, `label_utils.py`, `preprocessing.py` | `src/data/` |
| HP Tuning | `hyperparameter_tuning.py`, `config.py` | `src/training/` |
| Training | `train_model.py`, `config.py` | `src/training/` |
| CNN | `cnn_classifier.py` | `src/analysis/` |
| BiLSTM | `bilstm_classifier.py` | `src/analysis/` |
| Characterisation | `model_characterisation.py`, `disaster_vocab.py` | `src/analysis/` |
| Ensemble | `dynamic_ensemble.py`, `context_features.py` | `src/analysis/` |
| Confidence | `adaptive_confidence.py` | `src/analysis/` |
| Attribution | `attribution_filter.py`, `disaster_vocab.py` | `src/analysis/` |
| Evaluation | `evaluation.py` | `src/evaluation/` |
| Dashboard | `crisis_dashboard.py` | `src/app/` |

"""
Unified training script for all transformer models.
Supports class-weighted loss, training loss logging, and optional
hyperparameter loading from Optuna tuning results.

Usage (from src/training/):
    python train_model.py --model roberta --output_dir /path/to/output
    python train_model.py --model bert --output_dir /path/to/output
    python train_model.py --model bertweet --output_dir /path/to/output
    python train_model.py --model xtremedistil --output_dir /path/to/output

For Colab/Kaggle, import and call train_single_model().
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
    set_seed,
)
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from config import (
    NUM_LABELS, SEED, TRAINING_ARGS, MODEL_CONFIGS,
    DATA_DIR, TRAIN_FILE, VAL_FILE, TEST_FILE, LABEL_MAPPING_FILE,
)


# ── Weighted Trainer ─────────────────────────────────────────────────────────

class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy loss."""

    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(
                class_weights, dtype=torch.float32
            )
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weight = self.class_weights.to(device=logits.device, dtype=logits.dtype)
            loss_fct = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ── Training Loss Callback ───────────────────────────────────────────────────

class TrainingLossCallback(TrainerCallback):
    """
    Captures training loss, validation loss, and validation Macro F1
    at the end of each epoch. Saves a training_history.json file after
    training completes.
    """

    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_macro_f1": [],
        }
        self._current_train_loss = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Capture training loss from logs emitted during training."""
        if logs is not None and "loss" in logs:
            self._current_train_loss = logs["loss"]

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Capture validation metrics after each evaluation (end of epoch)."""
        if metrics is not None:
            epoch_num = int(state.epoch) if state.epoch else len(self.history["epoch"]) + 1
            self.history["epoch"].append(epoch_num)
            self.history["train_loss"].append(
                self._current_train_loss if self._current_train_loss is not None else 0.0
            )
            self.history["val_loss"].append(metrics.get("eval_loss", 0.0))
            self.history["val_macro_f1"].append(metrics.get("eval_macro_f1", 0.0))

    def on_train_end(self, args, state, control, **kwargs):
        """Save the full training history to JSON after training completes."""
        os.makedirs(self.output_dir, exist_ok=True)
        history_path = os.path.join(self.output_dir, "training_history.json")
        # Convert to serializable types
        serializable = {
            k: [float(v) for v in vals] for k, vals in self.history.items()
        }
        with open(history_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"  Training history saved to {history_path}")


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    """Compute macro F1 and accuracy for evaluation."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {"macro_f1": macro_f1, "accuracy": acc}


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_data():
    """Load train/val/test datasets from processed parquet files."""
    dataset = load_dataset(
        "parquet",
        data_files={
            "train": TRAIN_FILE,
            "validation": VAL_FILE,
            "test": TEST_FILE,
        },
    )
    return dataset


def load_label_mapping():
    """Load the label mapping."""
    with open(LABEL_MAPPING_FILE) as f:
        mapping = json.load(f)
    label2id = mapping["label2id"]
    id2label = {int(k): v for k, v in mapping["id2label"].items()}
    return label2id, id2label


def tokenize_dataset(dataset, tokenizer, max_length, model_key=None):
    """
    Tokenize all splits of a dataset.

    Note: BERTweet expects tweets normalised with URLs replaced by HTTPURL
    and mentions replaced by @USER. Our preprocessing removes these entirely,
    which is compatible — BERTweet's tokenizer handles cleaned text correctly.
    """
    def tokenize_fn(examples):
        encoded = tokenizer(
            examples["tweet_text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        return encoded

    return dataset.map(tokenize_fn, batched=True)


def compute_class_weights_from_labels(labels, num_classes):
    """Compute class weights using sklearn's balanced strategy."""
    labels_array = np.array(labels)
    classes = np.unique(labels_array)
    weights = compute_class_weight("balanced", classes=classes, y=labels_array)
    # Normalize so they sum to num_classes (standard normalization)
    weights = weights / weights.sum() * num_classes
    return weights.tolist()


def load_best_hyperparams(model_key):
    """
    Load Optuna-tuned best hyperparameters for a model if available.

    Returns dict of hyperparams or None if no tuning results exist.
    """
    hp_path = os.path.join(DATA_DIR, f"best_hyperparams_{model_key}.json")
    if os.path.exists(hp_path):
        with open(hp_path, "r") as f:
            params = json.load(f)
        print(f"  Loaded tuned hyperparameters from {hp_path}")
        return params
    return None


# ── Training ─────────────────────────────────────────────────────────────────

def train_single_model(
    model_key,
    seed=SEED,
    output_dir=".",
    dataset=None,
    label2id=None,
    id2label=None,
):
    """
    Train a single model with seed=42.

    Args:
        model_key: One of 'roberta', 'deberta', 'electra', 'bert', 'bertweet', 'xtremedistil'
        seed: Random seed (default: 42)
        output_dir: Base output directory
        dataset: Pre-loaded dataset (optional, will load if None)
        label2id: Label mapping (optional, will load if None)
        id2label: Label mapping (optional, will load if None)

    Returns:
        dict with training results and predictions
    """
    set_seed(seed)

    # Load data and labels if not provided
    if dataset is None:
        dataset = load_data()
    if label2id is None or id2label is None:
        label2id, id2label = load_label_mapping()

    # Model config
    model_config = MODEL_CONFIGS[model_key]
    model_name = model_config["model_name"]
    max_length = model_config["max_length"]

    print(f"\n{'='*70}")
    print(f"Training {model_key} (seed={seed})")
    print(f"  Model: {model_name}")
    print(f"  Max length: {max_length}")
    print(f"{'='*70}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize
    tokenized = tokenize_dataset(dataset, tokenizer, max_length, model_key=model_key)

    # Load pre-computed class weights if available, otherwise compute from training labels
    weights_path = os.path.join(DATA_DIR, "class_weights.pt")
    if os.path.exists(weights_path):
        class_weights = torch.load(weights_path).tolist()
        print(f"  Loaded pre-computed class weights: {[f'{w:.4f}' for w in class_weights]}")
    else:
        train_labels = tokenized["train"]["label"]
        class_weights = compute_class_weights_from_labels(train_labels, NUM_LABELS)
        print(f"  Computed class weights: {[f'{w:.4f}' for w in class_weights]}")

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        label2id=label2id,
        id2label=id2label,
    )

    # Output directory for this specific run
    run_output_dir = os.path.join(output_dir, model_key)

    # Build training arguments — start with defaults, then override with tuned params
    train_kwargs = dict(TRAINING_ARGS)

    # Check for Optuna-tuned hyperparameters
    best_hp = load_best_hyperparams(model_key)
    if best_hp is not None:
        for key in ["learning_rate", "warmup_ratio", "weight_decay", "per_device_train_batch_size"]:
            if key in best_hp:
                train_kwargs[key] = best_hp[key]
                print(f"  Using tuned {key}: {best_hp[key]}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=run_output_dir,
        seed=seed,
        **train_kwargs,
    )

    # Create training loss callback
    loss_callback = TrainingLossCallback(output_dir=run_output_dir)

    # Trainer with class-weighted loss
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2),
            loss_callback,
        ],
    )

    # Train
    train_result = trainer.train()
    print(f"  Training complete. Metrics: {train_result.metrics}")

    # Evaluate on validation set
    val_metrics = trainer.evaluate(tokenized["validation"])
    print(f"  Validation: {val_metrics}")

    # Evaluate on test set
    test_metrics = trainer.evaluate(tokenized["test"])
    print(f"  Test: {test_metrics}")

    # Get predictions on validation and test sets
    val_output = trainer.predict(tokenized["validation"])
    test_output = trainer.predict(tokenized["test"])

    results = {
        "model_key": model_key,
        "seed": seed,
        "train_metrics": train_result.metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_predictions": {
            "preds": np.argmax(val_output.predictions, axis=-1),
            "labels": val_output.label_ids,
            "probs": torch.softmax(
                torch.tensor(val_output.predictions), dim=-1
            ).numpy(),
        },
        "test_predictions": {
            "preds": np.argmax(test_output.predictions, axis=-1),
            "labels": test_output.label_ids,
            "probs": torch.softmax(
                torch.tensor(test_output.predictions), dim=-1
            ).numpy(),
        },
    }

    # Save predictions
    pred_dir = os.path.join(run_output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    np.savez(
        os.path.join(pred_dir, "val_predictions.npz"),
        preds=results["val_predictions"]["preds"],
        labels=results["val_predictions"]["labels"],
        probs=results["val_predictions"]["probs"],
    )
    np.savez(
        os.path.join(pred_dir, "test_predictions.npz"),
        preds=results["test_predictions"]["preds"],
        labels=results["test_predictions"]["labels"],
        probs=results["test_predictions"]["probs"],
    )

    # Save model
    trainer.save_model(os.path.join(run_output_dir, "best_model"))
    tokenizer.save_pretrained(os.path.join(run_output_dir, "best_model"))

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a transformer model")
    parser.add_argument(
        "--model", type=str, required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Which model to train",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Base output directory for model checkpoints and predictions",
    )
    args = parser.parse_args()

    train_single_model(
        model_key=args.model,
        seed=SEED,
        output_dir=args.output_dir,
    )

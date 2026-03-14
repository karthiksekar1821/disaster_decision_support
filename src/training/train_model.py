"""
Unified training script for all three transformer models.
Supports class-weighted loss, multi-seed training, and consistent hyperparameters.

Usage (from src/training/):
    python train_model.py --model roberta --output_dir /path/to/output
    python train_model.py --model deberta --output_dir /path/to/output
    python train_model.py --model electra --output_dir /path/to/output

For Colab/Kaggle, import and call train_single_model() or train_multi_seed().
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
    EarlyStoppingCallback,
    set_seed,
)
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from config import (
    NUM_LABELS, SEEDS, TRAINING_ARGS, MODEL_CONFIGS,
    TRAIN_FILE, VAL_FILE, TEST_FILE, LABEL_MAPPING_FILE,
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
            weight = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


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
    """Load the 5-class label mapping."""
    with open(LABEL_MAPPING_FILE) as f:
        mapping = json.load(f)
    label2id = mapping["label2id"]
    id2label = {int(k): v for k, v in mapping["id2label"].items()}
    return label2id, id2label


def tokenize_dataset(dataset, tokenizer, max_length):
    """Tokenize all splits of a dataset."""
    def tokenize_fn(examples):
        return tokenizer(
            examples["tweet_text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
    return dataset.map(tokenize_fn, batched=True)


def compute_class_weights_from_labels(labels, num_classes):
    """Compute class weights using sklearn's balanced strategy."""
    classes = np.arange(num_classes)
    weights = compute_class_weight("balanced", classes=classes, y=np.array(labels))
    # Normalize so they sum to num_classes (standard normalization)
    weights = weights / weights.sum() * num_classes
    return weights.tolist()


# ── Training ─────────────────────────────────────────────────────────────────

def train_single_model(
    model_key,
    seed,
    output_dir,
    dataset=None,
    label2id=None,
    id2label=None,
):
    """
    Train a single model with a single seed.

    Args:
        model_key: One of 'roberta', 'deberta', 'electra'
        seed: Random seed
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
    tokenized = tokenize_dataset(dataset, tokenizer, max_length)

    # Compute class weights from training labels
    train_labels = tokenized["train"]["label"]
    class_weights = compute_class_weights_from_labels(train_labels, NUM_LABELS)
    print(f"  Class weights: {[f'{w:.4f}' for w in class_weights]}")

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        label2id=label2id,
        id2label=id2label,
    )

    # Output directory for this specific run
    run_output_dir = os.path.join(output_dir, model_key, f"seed_{seed}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=run_output_dir,
        seed=seed,
        **TRAINING_ARGS,
    )

    # Trainer with class-weighted loss
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
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


# ── Multi-Seed Training ──────────────────────────────────────────────────────

def train_multi_seed(model_key, output_dir, seeds=None):
    """
    Train a model across multiple seeds and aggregate results.

    Returns:
        dict with per-seed results and aggregated metrics (mean ± std)
    """
    if seeds is None:
        seeds = SEEDS

    # Load data and labels once
    dataset = load_data()
    label2id, id2label = load_label_mapping()

    all_results = []
    for seed in seeds:
        result = train_single_model(
            model_key=model_key,
            seed=seed,
            output_dir=output_dir,
            dataset=dataset,
            label2id=label2id,
            id2label=id2label,
        )
        all_results.append(result)

    # Aggregate metrics
    test_f1s = [r["test_metrics"]["eval_macro_f1"] for r in all_results]
    test_accs = [r["test_metrics"]["eval_accuracy"] for r in all_results]
    val_f1s = [r["val_metrics"]["eval_macro_f1"] for r in all_results]
    val_accs = [r["val_metrics"]["eval_accuracy"] for r in all_results]

    summary = {
        "model_key": model_key,
        "seeds": seeds,
        "test_macro_f1_mean": float(np.mean(test_f1s)),
        "test_macro_f1_std": float(np.std(test_f1s)),
        "test_accuracy_mean": float(np.mean(test_accs)),
        "test_accuracy_std": float(np.std(test_accs)),
        "val_macro_f1_mean": float(np.mean(val_f1s)),
        "val_macro_f1_std": float(np.std(val_f1s)),
        "val_accuracy_mean": float(np.mean(val_accs)),
        "val_accuracy_std": float(np.std(val_accs)),
    }

    print(f"\n{'='*70}")
    print(f"MULTI-SEED SUMMARY FOR {model_key.upper()}")
    print(f"{'='*70}")
    print(f"  Test Macro F1:  {summary['test_macro_f1_mean']:.4f} ± {summary['test_macro_f1_std']:.4f}")
    print(f"  Test Accuracy:  {summary['test_accuracy_mean']:.4f} ± {summary['test_accuracy_std']:.4f}")
    print(f"  Val Macro F1:   {summary['val_macro_f1_mean']:.4f} ± {summary['val_macro_f1_std']:.4f}")
    print(f"  Val Accuracy:   {summary['val_accuracy_mean']:.4f} ± {summary['val_accuracy_std']:.4f}")

    # Save summary
    summary_path = os.path.join(output_dir, model_key, "multi_seed_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    return {"per_seed": all_results, "summary": summary}


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a transformer model")
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["roberta", "deberta", "electra"],
        help="Which model to train"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Base output directory for model checkpoints and predictions"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=SEEDS,
        help="Random seeds for multi-seed training"
    )
    parser.add_argument(
        "--single_seed", type=int, default=None,
        help="Train with a single seed only (overrides --seeds)"
    )
    args = parser.parse_args()

    if args.single_seed is not None:
        train_single_model(
            model_key=args.model,
            seed=args.single_seed,
            output_dir=args.output_dir,
        )
    else:
        train_multi_seed(
            model_key=args.model,
            output_dir=args.output_dir,
            seeds=args.seeds,
        )

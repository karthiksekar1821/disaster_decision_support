"""
Optuna-based Bayesian Hyperparameter Tuning for each transformer model.

For each model, runs 20 trials using the TPE sampler optimising for
validation Macro F1. Each trial trains on 30% of training data for
3 epochs maximum.

Usage (from Kaggle/Colab):
    from hyperparameter_tuning import run_hyperparameter_tuning
    run_hyperparameter_tuning("roberta", output_dir="/path/to/output")
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import optuna
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
    NUM_LABELS, SEED, TRAINING_ARGS, MODEL_CONFIGS,
    DATA_DIR, TRAIN_FILE, VAL_FILE, TEST_FILE, LABEL_MAPPING_FILE,
)
from train_model import (
    WeightedTrainer,
    compute_metrics,
    load_data,
    load_label_mapping,
    tokenize_dataset,
    compute_class_weights_from_labels,
)


def create_objective(model_key, dataset, label2id, id2label, class_weights, output_dir):
    """
    Create an Optuna objective function for a specific model.

    Each trial:
    - Samples hyperparameters from defined search space
    - Trains on 30% of training data for up to 3 epochs
    - Reports validation Macro F1
    """
    model_config = MODEL_CONFIGS[model_key]
    model_name = model_config["model_name"]
    max_length = model_config["max_length"]

    # Tokenize once (reused across trials)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized = tokenize_dataset(dataset, tokenizer, max_length, model_key=model_key)

    # Subsample 30% of training data for faster trials
    train_size = len(tokenized["train"])
    subset_size = int(train_size * 0.3)

    # Use a deterministic shuffle for reproducibility
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(train_size)[:subset_size].tolist()
    train_subset = tokenized["train"].select(indices)

    def objective(trial):
        set_seed(SEED)

        # Sample hyperparameters
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
        warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
        weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
        batch_size = trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32])

        # Create model fresh each trial
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=NUM_LABELS,
            label2id=label2id,
            id2label=id2label,
        )

        trial_output_dir = os.path.join(output_dir, f"hp_tuning/{model_key}/trial_{trial.number}")

        training_args = TrainingArguments(
            output_dir=trial_output_dir,
            seed=SEED,
            num_train_epochs=3,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=32,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=1,
            fp16=torch.cuda.is_available(),
            report_to="none",
        )

        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_subset,
            eval_dataset=tokenized["validation"],
            compute_metrics=compute_metrics,
        )

        trainer.train()

        # Evaluate on validation set
        val_metrics = trainer.evaluate(tokenized["validation"])
        macro_f1 = val_metrics.get("eval_macro_f1", 0.0)

        # Clean up trial checkpoint to save disk space
        import shutil
        if os.path.exists(trial_output_dir):
            shutil.rmtree(trial_output_dir, ignore_errors=True)

        return macro_f1

    return objective


def run_hyperparameter_tuning(model_key, output_dir, n_trials=20):
    """
    Run Optuna hyperparameter tuning for a single model.

    Args:
        model_key: One of the keys in MODEL_CONFIGS
        output_dir: Base output directory
        n_trials: Number of trials (default: 20)

    Returns:
        dict with best hyperparameters
    """
    print(f"\n{'='*70}")
    print(f"HYPERPARAMETER TUNING: {model_key.upper()}")
    print(f"  Trials: {n_trials}")
    print(f"  Sampler: TPE")
    print(f"  Objective: Validation Macro F1")
    print(f"{'='*70}")

    # Load data
    dataset = load_data()
    label2id, id2label = load_label_mapping()

    # Compute class weights
    weights_path = os.path.join(DATA_DIR, "class_weights.pt")
    if os.path.exists(weights_path):
        class_weights = torch.load(weights_path).tolist()
    else:
        train_labels = dataset["train"]["label"]
        class_weights = compute_class_weights_from_labels(train_labels, NUM_LABELS)

    # Create Optuna study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        study_name=f"hp_tuning_{model_key}",
    )

    # Create objective
    objective = create_objective(
        model_key=model_key,
        dataset=dataset,
        label2id=label2id,
        id2label=id2label,
        class_weights=class_weights,
        output_dir=output_dir,
    )

    # Run optimization
    study.optimize(objective, n_trials=n_trials)

    # Best results
    best_params = study.best_params
    best_value = study.best_value

    print(f"\n{'='*70}")
    print(f"BEST RESULTS FOR {model_key.upper()}")
    print(f"  Best Macro F1: {best_value:.4f}")
    print(f"  Best params: {json.dumps(best_params, indent=2)}")
    print(f"{'='*70}")

    # Save best hyperparameters
    hp_path = os.path.join(DATA_DIR, f"best_hyperparams_{model_key}.json")
    os.makedirs(os.path.dirname(hp_path), exist_ok=True)
    with open(hp_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"  Saved best hyperparams to {hp_path}")

    return best_params


def run_all_hyperparameter_tuning(output_dir, n_trials=20):
    """Run hyperparameter tuning for all models."""
    all_best_params = {}
    for model_key in MODEL_CONFIGS:
        best_params = run_hyperparameter_tuning(
            model_key=model_key,
            output_dir=output_dir,
            n_trials=n_trials,
        )
        all_best_params[model_key] = best_params
    return all_best_params


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hyperparameter tuning")
    parser.add_argument(
        "--model", type=str, default=None,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to tune (all if not specified)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Base output directory",
    )
    parser.add_argument(
        "--n_trials", type=int, default=20,
        help="Number of Optuna trials per model",
    )
    args = parser.parse_args()

    if args.model:
        run_hyperparameter_tuning(args.model, args.output_dir, args.n_trials)
    else:
        run_all_hyperparameter_tuning(args.output_dir, args.n_trials)

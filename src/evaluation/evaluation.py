"""
Comprehensive Evaluation Script.

Covers:
1. Individual model results (all 8 models: 6 transformers + CNN + BiLSTM)
2. Simple baselines (TF-IDF + SVM, majority class)
3. Dynamic ensemble results (Novelty 1)
4. Class-adaptive confidence (Novelty 2)
5. Attribution reliability (Novelty 3)
6. Statistical significance (McNemar's test)
7. Confusion matrices
8. Training curves (from training_history.json)
9. Model style performance heatmap
10. Error analysis

Usage: Import and run functions from a Colab notebook.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    confusion_matrix,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from scipy.stats import chi2


# ── McNemar's Test ───────────────────────────────────────────────────────────

def mcnemar_test(preds_a, preds_b, labels):
    """
    McNemar's test for comparing two classifiers.

    Returns:
        chi2_stat: test statistic
        p_value: p-value
        summary: dict with b, c counts and results
    """
    # b = A correct, B incorrect
    # c = A incorrect, B correct
    a_correct = (preds_a == labels)
    b_correct = (preds_b == labels)

    b = int(np.sum(a_correct & ~b_correct))  # A right, B wrong
    c = int(np.sum(~a_correct & b_correct))  # A wrong, B right

    if b + c == 0:
        return 0.0, 1.0, {"b": b, "c": c, "significant": False}

    # McNemar's test with continuity correction
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - chi2.cdf(chi2_stat, df=1)

    return chi2_stat, p_value, {
        "b": b, "c": c,
        "chi2": float(chi2_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
    }


# ── Simple Baselines ─────────────────────────────────────────────────────────

def train_tfidf_svm_baseline(train_texts, train_labels, test_texts, test_labels):
    """
    Train a TF-IDF + LinearSVC baseline.

    Returns:
        results dict with predictions and metrics
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("svm", LinearSVC(max_iter=5000, random_state=42)),
    ])

    pipeline.fit(train_texts, train_labels)
    preds = pipeline.predict(test_texts)

    macro_f1 = f1_score(test_labels, preds, average="macro")
    acc = accuracy_score(test_labels, preds)

    print(f"\nTF-IDF + SVM Baseline:")
    print(f"  Macro F1:  {macro_f1:.4f}")
    print(f"  Accuracy:  {acc:.4f}")

    return {
        "preds": preds,
        "macro_f1": float(macro_f1),
        "accuracy": float(acc),
    }


def majority_class_baseline(train_labels, test_labels):
    """
    Majority class baseline.

    Returns:
        results dict
    """
    from collections import Counter
    majority_class = Counter(train_labels).most_common(1)[0][0]
    preds = np.full(len(test_labels), majority_class)

    macro_f1 = f1_score(test_labels, preds, average="macro")
    acc = accuracy_score(test_labels, preds)

    print(f"\nMajority Class Baseline (class={majority_class}):")
    print(f"  Macro F1:  {macro_f1:.4f}")
    print(f"  Accuracy:  {acc:.4f}")

    return {
        "majority_class": int(majority_class),
        "preds": preds,
        "macro_f1": float(macro_f1),
        "accuracy": float(acc),
    }


# ── Visualization ────────────────────────────────────────────────────────────

def plot_confusion_matrix(labels, preds, class_names, title, save_path=None):
    """Plot a normalized confusion matrix."""
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, ax=ax, vmin=0, vmax=1,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_class_f1_comparison(
    model_results, class_names, title, save_path=None,
):
    """Plot per-class F1 comparison across models."""
    n_classes = len(class_names)
    n_models = len(model_results)
    x = np.arange(n_classes)
    width = 0.8 / max(n_models, 1)

    fig, ax = plt.subplots(figsize=(max(14, n_models * 2), 7))

    for i, (name, results) in enumerate(model_results.items()):
        f1_vals = f1_score(
            results["labels"], results["preds"],
            average=None, labels=list(range(n_classes)),
        )
        ax.bar(
            x + i * width, f1_vals, width,
            label=name, alpha=0.85,
        )

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_macro_f1_summary(model_results, title, save_path=None):
    """Plot macro F1 comparison bar chart."""
    names = list(model_results.keys())
    f1s = [
        f1_score(r["labels"], r["preds"], average="macro")
        for r in model_results.values()
    ]

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.2), 6))
    bars = ax.bar(names, f1s, alpha=0.85, width=0.5)

    for bar, val in zip(bars, f1s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_ylabel("Macro F1 Score", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Training Curves ──────────────────────────────────────────────────────────

def plot_training_curves(history_dir, model_name, save_path=None):
    """
    Plot training curves for a single model.

    Reads training_history.json and plots:
    - Train loss and val loss on the primary y-axis
    - Val Macro F1 on the secondary y-axis

    Args:
        history_dir: directory containing training_history.json
        model_name: display name of the model
        save_path: path to save the plot
    """
    history_path = os.path.join(history_dir, "training_history.json")
    if not os.path.exists(history_path):
        print(f"  Warning: {history_path} not found, skipping training curves for {model_name}")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = history["epoch"]
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    val_macro_f1 = history["val_macro_f1"]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Loss on primary axis
    ax1.plot(epochs, train_loss, "b-o", label="Train Loss", markersize=5)
    ax1.plot(epochs, val_loss, "r-o", label="Val Loss", markersize=5)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12, color="black")
    ax1.tick_params(axis="y")
    ax1.legend(loc="upper left", fontsize=10)

    # Macro F1 on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(epochs, val_macro_f1, "g-s", label="Val Macro F1", markersize=5)
    ax2.set_ylabel("Macro F1", fontsize=12, color="green")
    ax2.tick_params(axis="y", labelcolor="green")
    ax2.legend(loc="upper right", fontsize=10)

    ax1.set_title(f"Training Curves: {model_name}", fontsize=14, fontweight="bold")
    ax1.grid(axis="both", alpha=0.3)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_all_training_curves(output_dir, model_keys, save_dir):
    """Plot training curves for all models."""
    os.makedirs(save_dir, exist_ok=True)
    for model_key in model_keys:
        history_dir = os.path.join(output_dir, model_key)
        save_path = os.path.join(save_dir, f"training_curves_{model_key}.png")
        plot_training_curves(history_dir, model_key, save_path)


# ── Calibration Diagram ─────────────────────────────────────────────────────

def plot_calibration_diagram(model_results, n_bins=10, save_path=None):
    """Plot reliability/calibration diagram for all models."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for name, results in model_results.items():
        if "probs" not in results:
            continue

        probs = results["probs"]
        preds = results["preds"]
        labels = results["labels"]

        # Get predicted class probability for each sample
        predicted_probs = np.array([probs[i, preds[i]] for i in range(len(preds))])

        # Bin by predicted probability
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_accuracies = []
        bin_confidences = []

        for b in range(n_bins):
            lo, hi = bin_boundaries[b], bin_boundaries[b + 1]
            mask = (predicted_probs >= lo) & (predicted_probs < hi)
            if mask.sum() > 0:
                bin_acc = (preds[mask] == labels[mask]).mean()
                bin_conf = predicted_probs[mask].mean()
                bin_accuracies.append(bin_acc)
                bin_confidences.append(bin_conf)

        ax.plot(bin_confidences, bin_accuracies, "-o", label=name, markersize=4)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Diagram", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Confidence Distribution ─────────────────────────────────────────────────

def plot_confidence_distributions(model_results, save_path=None):
    """Plot confidence distribution for all models."""
    n_models = len(model_results)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (name, results) in enumerate(model_results.items()):
        ax = axes[idx]
        if "probs" not in results:
            ax.set_visible(False)
            continue

        probs = results["probs"]
        preds = results["preds"]
        max_probs = np.max(probs, axis=1)

        ax.hist(max_probs, bins=50, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.set_title(f"{name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Max Probability", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.axvline(np.mean(max_probs), color="red", linestyle="--",
                   label=f"Mean: {np.mean(max_probs):.3f}")
        ax.legend(fontsize=9)

    # Hide unused axes
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Confidence Distributions per Model", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Consolidated Results Table ───────────────────────────────────────────────

def print_consolidated_results(model_results, ensemble_results, class_names):
    """Print a consolidated results table for all models."""
    print(f"\n{'='*80}")
    print(f"CONSOLIDATED RESULTS TABLE")
    print(f"{'='*80}")
    print(f"{'Model':<15} {'Macro F1':>10} {'Accuracy':>10} {'Per-Class F1'}")
    print(f"{'-'*80}")

    all_results = {**model_results, "Ensemble": ensemble_results}

    for name, results in all_results.items():
        f1 = f1_score(results["labels"], results["preds"], average="macro")
        acc = accuracy_score(results["labels"], results["preds"])
        per_class = f1_score(
            results["labels"], results["preds"],
            average=None, labels=list(range(len(class_names))),
        )
        per_class_str = " ".join([f"{v:.3f}" for v in per_class])
        print(f"{name:<15} {f1:>10.4f} {acc:>10.4f} {per_class_str}")

    print(f"{'='*80}")


# ── Comprehensive Report ─────────────────────────────────────────────────────

def run_full_evaluation(
    model_test_results,
    ensemble_test_results,
    selective_results,
    combined_results,
    train_texts,
    train_labels,
    test_texts,
    test_labels,
    class_names,
    output_dir,
    model_output_dir=None,
    style_performance_path=None,
):
    """
    Run the full evaluation pipeline and save all results.

    Args:
        model_test_results: dict of model_name -> {"preds", "labels", "probs"}
            (can include up to 8 models)
        ensemble_test_results: {"preds", "labels", "probs"}
        selective_results: results from adaptive_confidence
        combined_results: results from attribution_filter combined_abstention
        train_texts: list of training tweet texts
        train_labels: array of training labels
        test_texts: list of test tweet texts
        test_labels: array of test labels
        class_names: list of class name strings
        output_dir: directory to save results
        model_output_dir: base dir for model outputs (for training curves)
        style_performance_path: path to model_style_performance.json
    """
    os.makedirs(output_dir, exist_ok=True)

    print("="*70)
    print("COMPREHENSIVE EVALUATION REPORT")
    print("="*70)

    # 1. Individual model results
    print("\n1. INDIVIDUAL MODEL RESULTS")
    print("-"*40)
    for name, results in model_test_results.items():
        f1 = f1_score(results["labels"], results["preds"], average="macro")
        acc = accuracy_score(results["labels"], results["preds"])
        print(f"  {name}: Macro F1={f1:.4f}  Accuracy={acc:.4f}")

    # 2. Simple baselines
    print("\n2. BASELINES")
    print("-"*40)
    tfidf_results = train_tfidf_svm_baseline(
        train_texts, train_labels, test_texts, test_labels
    )
    majority_results = majority_class_baseline(train_labels, test_labels)

    # 3. Ensemble results
    print("\n3. DYNAMIC ENSEMBLE (NOVELTY 1)")
    print("-"*40)
    ens_f1 = f1_score(
        ensemble_test_results["labels"],
        ensemble_test_results["preds"],
        average="macro",
    )
    ens_acc = accuracy_score(
        ensemble_test_results["labels"],
        ensemble_test_results["preds"],
    )
    print(f"  Macro F1:  {ens_f1:.4f}")
    print(f"  Accuracy:  {ens_acc:.4f}")

    # 4. Selective prediction (Novelty 2)
    print("\n4. CLASS-ADAPTIVE SELECTIVE PREDICTION (NOVELTY 2)")
    print("-"*40)
    if selective_results:
        for k, v in selective_results.items():
            print(f"  {k}: {v}")

    # 5. Combined abstention (Novelty 2 + 3)
    print("\n5. COMBINED ABSTENTION (NOVELTY 2 + 3)")
    print("-"*40)
    if combined_results:
        for k, v in combined_results.items():
            print(f"  {k}: {v}")

    # 6. Statistical significance
    print("\n6. STATISTICAL SIGNIFICANCE (McNemar's Test)")
    print("-"*40)
    model_names = list(model_test_results.keys())
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            name_a = model_names[i]
            name_b = model_names[j]
            chi2_stat, p_val, summary = mcnemar_test(
                model_test_results[name_a]["preds"],
                model_test_results[name_b]["preds"],
                test_labels,
            )
            sig = "YES" if summary["significant"] else "NO"
            print(f"  {name_a} vs {name_b}: "
                  f"chi2={chi2_stat:.4f}, p={p_val:.4f}, significant={sig}")

    # Also test ensemble vs best individual
    best_model_name = max(
        model_test_results.keys(),
        key=lambda n: f1_score(
            model_test_results[n]["labels"],
            model_test_results[n]["preds"],
            average="macro",
        ),
    )
    chi2_stat, p_val, summary = mcnemar_test(
        ensemble_test_results["preds"],
        model_test_results[best_model_name]["preds"],
        test_labels,
    )
    sig = "YES" if summary["significant"] else "NO"
    print(f"  Ensemble vs {best_model_name}: "
          f"chi2={chi2_stat:.4f}, p={p_val:.4f}, significant={sig}")

    # 7. Confusion matrices — for each model + ensemble
    print("\n7. CONFUSION MATRICES")
    print("-"*40)
    for name, results in model_test_results.items():
        plot_confusion_matrix(
            results["labels"], results["preds"], class_names,
            f"{name} Confusion Matrix",
            save_path=os.path.join(output_dir, f"confusion_matrix_{name.lower().replace(' ', '_')}.png"),
        )

    plot_confusion_matrix(
        ensemble_test_results["labels"],
        ensemble_test_results["preds"],
        class_names,
        "Dynamic Ensemble Confusion Matrix",
        save_path=os.path.join(output_dir, "ensemble_confusion_matrix.png"),
    )

    # 8. Per-class F1 comparison
    all_results = {**model_test_results}
    all_results["Ensemble"] = ensemble_test_results
    plot_per_class_f1_comparison(
        all_results, class_names,
        "Per-Class F1: All Models vs Dynamic Ensemble",
        save_path=os.path.join(output_dir, "per_class_f1_comparison.png"),
    )

    # 9. Macro F1 summary
    all_results_with_baselines = {
        "Majority": {"preds": majority_results["preds"], "labels": test_labels},
        "TF-IDF+SVM": {"preds": tfidf_results["preds"], "labels": test_labels},
        **model_test_results,
        "Ensemble": ensemble_test_results,
    }
    plot_macro_f1_summary(
        all_results_with_baselines,
        "Macro F1 Comparison: Baselines vs Models vs Ensemble",
        save_path=os.path.join(output_dir, "macro_f1_summary.png"),
    )

    # 10. Calibration diagram
    models_with_probs = {
        name: r for name, r in model_test_results.items() if "probs" in r
    }
    if ensemble_test_results.get("probs") is not None:
        models_with_probs["Ensemble"] = ensemble_test_results
    plot_calibration_diagram(
        models_with_probs,
        save_path=os.path.join(output_dir, "calibration_diagram.png"),
    )

    # 11. Confidence distributions
    plot_confidence_distributions(
        models_with_probs,
        save_path=os.path.join(output_dir, "confidence_distributions.png"),
    )

    # 12. Training curves (if model_output_dir provided)
    if model_output_dir:
        print("\n8. TRAINING CURVES")
        print("-"*40)
        transformer_keys = ["roberta", "deberta", "electra", "bert", "xlnet", "xtremedistil"]
        plot_all_training_curves(model_output_dir, transformer_keys, output_dir)
        print("  Training curve plots saved.")

    # 13. Model style performance heatmap
    if style_performance_path and os.path.exists(style_performance_path):
        print("\n9. MODEL STYLE PERFORMANCE HEATMAP")
        print("-"*40)
        try:
            from model_characterisation import plot_style_performance_heatmap, STYLE_CATEGORIES
            with open(style_performance_path, "r") as f:
                perf_matrix = json.load(f)
            plot_style_performance_heatmap(
                perf_matrix,
                list(perf_matrix.keys()),
                save_path=os.path.join(output_dir, "model_style_heatmap.png"),
            )
        except Exception as e:
            print(f"  Warning: Could not generate style heatmap: {e}")

    # 14. Consolidated results table
    print_consolidated_results(model_test_results, ensemble_test_results, class_names)

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)

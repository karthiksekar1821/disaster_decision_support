"""
Comprehensive Evaluation Script.

Covers:
1. Individual model results (per-model, per-seed, mean ± std)
2. Simple baselines (TF-IDF + SVM, majority class)
3. Dynamic ensemble results (Novelty 1)
4. Class-adaptive confidence (Novelty 2)
5. Attribution reliability (Novelty 3)
6. Statistical significance (McNemar's test)
7. Confusion matrices
8. Error analysis

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
    plt.show()


def plot_per_class_f1_comparison(
    model_results, class_names, title, save_path=None,
):
    """Plot per-class F1 comparison across models."""
    n_classes = len(class_names)
    x = np.arange(n_classes)
    width = 0.8 / len(model_results)

    fig, ax = plt.subplots(figsize=(14, 6))

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
    ax.set_xticks(x + width * (len(model_results) - 1) / 2)
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_macro_f1_summary(model_results, title, save_path=None):
    """Plot macro F1 comparison bar chart."""
    names = list(model_results.keys())
    f1s = [
        f1_score(r["labels"], r["preds"], average="macro")
        for r in model_results.values()
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, f1s, alpha=0.85, width=0.5)

    for bar, val in zip(bars, f1s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_ylabel("Macro F1 Score", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


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
):
    """
    Run the full evaluation pipeline and save all results.

    Args:
        model_test_results: dict of model_name -> {"preds", "labels", "probs"}
        ensemble_test_results: {"preds", "labels", "probs"}
        selective_results: results from adaptive_confidence
        combined_results: results from attribution_filter combined_abstention
        train_texts: list of training tweet texts
        train_labels: array of training labels
        test_texts: list of test tweet texts
        test_labels: array of test labels
        class_names: list of class name strings
        output_dir: directory to save results
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

    # 7. Confusion matrices
    print("\n7. CONFUSION MATRICES")
    print("-"*40)
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
        "Per-Class F1: Individual Models vs Dynamic Ensemble",
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

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)

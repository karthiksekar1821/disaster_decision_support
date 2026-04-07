"""
Class-Adaptive Confidence Thresholds (Novelty 2).

Instead of one global threshold, compute a separate optimal threshold
per class. Some classes have high model confidence and can use a higher
threshold. Others have low confidence and need a lower one.

Thresholds are swept on the VALIDATION set independently per class.
The result is a per-class abstention policy.
"""

import numpy as np
from sklearn.metrics import f1_score


def sweep_per_class_thresholds(
    probs,
    preds,
    labels,
    num_classes,
    threshold_range=None,
    target_coverage=None,
    alpha=0.7,
    min_threshold=0.3,
    min_coverage=0.5,
):
    """
    Sweep thresholds per class on validation data.

    For each class c:
      - Consider only samples predicted as class c
      - Sweep threshold t_c from min_threshold to 1.0
      - Accept prediction only if prob[c] >= t_c
      - Only consider thresholds where per-class coverage >= min_coverage
      - Choose t_c that maximizes: alpha * F1_c + (1-alpha) * coverage_c

    Args:
        probs: (N, C) softmax probabilities
        preds: (N,) predicted class indices
        labels: (N,) true class indices
        num_classes: number of classes
        threshold_range: (start, stop, step) for threshold sweep
        target_coverage: if set, prefer thresholds that achieve this coverage
        alpha: weight for F1 vs coverage in the objective (default 0.7).
               Higher alpha prioritises F1 improvement; lower values keep
               more coverage.
        min_threshold: minimum threshold floor for any class (default 0.3).
               Prevents trivially-zero thresholds that accept everything.
        min_coverage: minimum per-class coverage required (default 0.5).
               Thresholds that reject more than half the predictions for
               a class are skipped to avoid excessive abstention.

    Returns:
        per_class_thresholds: dict mapping class_idx -> optimal threshold
        per_class_stats: dict mapping class_idx -> stats dict
    """
    if threshold_range is None:
        thresholds = np.arange(0.0, 1.01, 0.01)
    else:
        thresholds = np.arange(*threshold_range)

    per_class_thresholds = {}
    per_class_stats = {}

    for c in range(num_classes):
        # Enforce minimum threshold floor
        best_threshold = float(min_threshold)
        best_score = -1.0
        best_stats = None

        for t in thresholds:
            # Skip thresholds below the floor
            if t < min_threshold:
                continue

            # Samples predicted as class c with confidence >= t
            mask = (preds == c) & (probs[:, c] >= t)
            accepted = mask.sum()

            # Total samples predicted as class c
            total_predicted_c = (preds == c).sum()

            if accepted == 0:
                continue

            # Coverage for this class
            coverage_c = accepted / max(total_predicted_c, 1)

            # Skip if coverage drops below minimum
            if coverage_c < min_coverage:
                continue

            # F1 among accepted predictions for this class
            # We compute binary F1: did we correctly predict class c?
            accepted_preds = preds[mask]
            accepted_labels = labels[mask]

            # Compute how many of the accepted are correct
            correct = (accepted_preds == accepted_labels).sum()
            precision_c = correct / max(accepted, 1)

            # Recall: of all true class c samples, how many did we
            # correctly predict AND accept?
            true_c_mask = labels == c
            true_c_count = true_c_mask.sum()

            if true_c_count == 0:
                recall_c = 0.0
            else:
                correctly_accepted_c = (
                    (preds == c) & (labels == c) & (probs[:, c] >= t)
                ).sum()
                recall_c = correctly_accepted_c / true_c_count

            # F1 for this class
            if precision_c + recall_c > 0:
                f1_c = 2 * precision_c * recall_c / (precision_c + recall_c)
            else:
                f1_c = 0.0

            # Objective: balance F1 and coverage
            score = alpha * f1_c + (1 - alpha) * coverage_c

            if score > best_score:
                best_score = score
                best_threshold = float(t)
                best_stats = {
                    "threshold": float(t),
                    "f1": float(f1_c),
                    "precision": float(precision_c),
                    "recall": float(recall_c),
                    "coverage": float(coverage_c),
                    "accepted": int(accepted),
                    "total_predicted": int(total_predicted_c),
                    "objective_score": float(score),
                }

        # If no valid threshold was found (best_stats is None),
        # fall back to the minimum threshold and compute stats for it
        if best_stats is None:
            t = min_threshold
            mask = (preds == c) & (probs[:, c] >= t)
            accepted = mask.sum()
            total_predicted_c = (preds == c).sum()
            coverage_c = accepted / max(total_predicted_c, 1) if total_predicted_c > 0 else 0.0
            correct = ((preds[mask] == labels[mask]).sum()) if accepted > 0 else 0
            precision_c = correct / max(accepted, 1)
            true_c_count = (labels == c).sum()
            correctly_accepted_c = ((preds == c) & (labels == c) & (probs[:, c] >= t)).sum()
            recall_c = correctly_accepted_c / true_c_count if true_c_count > 0 else 0.0
            f1_c = 2 * precision_c * recall_c / (precision_c + recall_c) if (precision_c + recall_c) > 0 else 0.0
            best_stats = {
                "threshold": float(t),
                "f1": float(f1_c),
                "precision": float(precision_c),
                "recall": float(recall_c),
                "coverage": float(coverage_c),
                "accepted": int(accepted),
                "total_predicted": int(total_predicted_c),
                "objective_score": float(alpha * f1_c + (1 - alpha) * coverage_c),
            }

        per_class_thresholds[c] = best_threshold
        per_class_stats[c] = best_stats

    return per_class_thresholds, per_class_stats


def apply_per_class_thresholds(probs, preds, per_class_thresholds):
    """
    Apply per-class thresholds to predictions.

    Returns:
        accepted_mask: boolean array, True if prediction is accepted
    """
    N = len(preds)
    accepted = np.zeros(N, dtype=bool)

    for i in range(N):
        pred_class = preds[i]
        threshold = per_class_thresholds.get(pred_class, 0.0)
        if probs[i, pred_class] >= threshold:
            accepted[i] = True

    return accepted


def evaluate_selective_prediction(
    probs, preds, labels, per_class_thresholds, class_names,
):
    """
    Evaluate the selective prediction system.

    Returns:
        results dict with overall and per-class metrics
    """
    accepted_mask = apply_per_class_thresholds(probs, preds, per_class_thresholds)

    total = len(preds)
    accepted_count = accepted_mask.sum()
    overall_coverage = accepted_count / total

    # Metrics among accepted predictions
    if accepted_count > 0:
        accepted_preds = preds[accepted_mask]
        accepted_labels = labels[accepted_mask]
        macro_f1 = f1_score(accepted_labels, accepted_preds, average="macro")
        accuracy = (accepted_preds == accepted_labels).mean()
    else:
        macro_f1 = 0.0
        accuracy = 0.0

    # Full metrics (without selective prediction) for comparison
    full_macro_f1 = f1_score(labels, preds, average="macro")
    full_accuracy = (preds == labels).mean()

    print(f"\n{'='*60}")
    print(f"CLASS-ADAPTIVE SELECTIVE PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"  Per-class thresholds:")
    for c, t in sorted(per_class_thresholds.items()):
        name = class_names[c] if c < len(class_names) else f"class_{c}"
        print(f"    {name}: {t:.2f}")
    print(f"\n  Without selective prediction:")
    print(f"    Macro F1:  {full_macro_f1:.4f}")
    print(f"    Accuracy:  {full_accuracy:.4f}")
    print(f"\n  With class-adaptive thresholds:")
    print(f"    Coverage:  {overall_coverage:.4f} ({accepted_count}/{total})")
    print(f"    Macro F1:  {macro_f1:.4f}")
    print(f"    Accuracy:  {accuracy:.4f}")
    print(f"    F1 improvement: +{macro_f1 - full_macro_f1:.4f}")

    results = {
        "per_class_thresholds": {
            class_names[c]: t for c, t in per_class_thresholds.items()
        },
        "full": {"macro_f1": float(full_macro_f1), "accuracy": float(full_accuracy)},
        "selective": {
            "coverage": float(overall_coverage),
            "accepted_count": int(accepted_count),
            "total": int(total),
            "macro_f1": float(macro_f1),
            "accuracy": float(accuracy),
        },
    }

    return results, accepted_mask

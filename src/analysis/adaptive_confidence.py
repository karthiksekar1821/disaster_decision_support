"""
Class-Adaptive Confidence Thresholds (Novelty 2).

Instead of one global threshold, compute a separate optimal threshold
per class. Some classes have high model confidence and can use a higher
threshold. Others have low confidence and need a lower one.

Thresholds are swept on the VALIDATION set independently per class.
The result is a per-class abstention policy.
"""

import numpy as np
from sklearn.metrics import f1_score, accuracy_score


def sweep_per_class_thresholds(
    ensemble_probs,
    ensemble_preds,
    true_labels,
    class_names,
    alpha=0.8,
):
    """
    Sweep confidence thresholds independently per class.
    Finds the threshold that maximises alpha*F1 + (1-alpha)*coverage
    with minimum threshold of 0.5 and maximum coverage of 0.90 per class.
    """
    thresholds = np.arange(0.50, 0.96, 0.05)
    max_conf = np.max(ensemble_probs, axis=1)
    pred_classes = ensemble_preds

    per_class_thresholds = {}

    for class_idx, class_name in enumerate(class_names):
        best_threshold = 0.50
        best_score = -1

        for t in thresholds:
            # For this class, accepted = predictions for this class above threshold
            # OR predictions for other classes (we only abstain when THIS class
            # is predicted with low confidence)
            mask_this_class = pred_classes == class_idx
            mask_above_threshold = max_conf >= t

            # Abstain only on this class predictions below threshold
            accepted_mask = (~mask_this_class) | (mask_this_class & mask_above_threshold)

            coverage = accepted_mask.mean()

            # Skip if coverage too high (not abstaining enough) or too low
            if coverage > 0.92 or coverage < 0.50:
                continue

            # Compute F1 on accepted predictions
            accepted_preds = pred_classes[accepted_mask]
            accepted_labels = true_labels[accepted_mask]

            if len(accepted_preds) == 0:
                continue

            f1 = f1_score(accepted_labels, accepted_preds,
                         average="macro", zero_division=0)

            score = alpha * f1 + (1 - alpha) * coverage

            if score > best_score:
                best_score = score
                best_threshold = t

        per_class_thresholds[class_name] = float(best_threshold)

    return per_class_thresholds


def apply_per_class_thresholds(
    ensemble_probs,
    ensemble_preds,
    true_labels,
    per_class_thresholds,
    class_names,
):
    """
    Apply per-class thresholds. A prediction is accepted if its confidence
    meets the threshold for its predicted class. Otherwise abstained.
    """
    max_conf = np.max(ensemble_probs, axis=1)

    # Build threshold array aligned with predictions
    threshold_array = np.array([
        per_class_thresholds[class_names[pred]]
        for pred in ensemble_preds
    ])

    accepted_mask = max_conf >= threshold_array

    accepted_preds  = ensemble_preds[accepted_mask]
    accepted_labels = true_labels[accepted_mask]
    coverage        = accepted_mask.mean()
    accepted_count  = accepted_mask.sum()

    macro_f1 = f1_score(accepted_labels, accepted_preds,
                       average="macro", zero_division=0)
    accuracy = accuracy_score(accepted_labels, accepted_preds)

    return {
        "coverage": float(coverage),
        "accepted_count": int(accepted_count),
        "total": len(ensemble_preds),
        "macro_f1": float(macro_f1),
        "accuracy": float(accuracy),
        "accepted_mask": accepted_mask,
    }


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

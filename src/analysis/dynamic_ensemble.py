"""
Dynamic Context-Conditioned Ensemble (Novelty 1).

Instead of fixed ensemble weights, this module trains a meta-learner
that produces per-sample ensemble weights conditioned on each tweet's
event context, linguistic profile, tweet style, and per-model confidence gaps.

CRITICAL FIX: Meta-learner is trained on VALIDATION set and evaluated
on TEST set (not trained/evaluated on the same data).

Updated to support 8 models: 6 transformers + CNN + BiLSTM.
"""

import os
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, classification_report
import joblib

from context_features import extract_features_batch


NUM_MODELS = 8
MODEL_NAMES = [
    "RoBERTa", "DeBERTa", "ELECTRA",
    "BERT", "BERTweet", "XtremeDistil",
    "CNN", "BiLSTM",
]


def compute_confidence_gaps(model_probs_list):
    """
    Compute per-model confidence gap: max_prob - second_max_prob.

    Args:
        model_probs_list: list of arrays, each shape (N, num_classes)

    Returns:
        gaps: array of shape (N, num_models)
    """
    gaps = []
    for probs in model_probs_list:
        sorted_probs = np.sort(probs, axis=1)[:, ::-1]  # descending
        gap = sorted_probs[:, 0] - sorted_probs[:, 1]   # max - second max
        gaps.append(gap.reshape(-1, 1))
    return np.concatenate(gaps, axis=1)  # (N, num_models)


def build_meta_features(model_probs_list, tweet_texts, style_labels=None):
    """
    Build meta-learner input features.

    Args:
        model_probs_list: list of M arrays, each shape (N, num_classes)
            softmax probabilities from each model
        tweet_texts: list of N tweet strings
        style_labels: optional list of N style label strings

    Returns:
        feature_matrix: shape (N, M*num_classes + num_context_features + style_features + gap_features)
        feature_names: list of feature names
    """
    num_models = len(model_probs_list)
    num_classes = model_probs_list[0].shape[1]
    model_names = MODEL_NAMES[:num_models]

    # 1. Concatenate softmax probabilities from all models
    prob_features = np.concatenate(model_probs_list, axis=1)  # (N, M*C)
    prob_names = [
        f"{name}_prob_c{c}"
        for name in model_names
        for c in range(num_classes)
    ]

    # 2. Extract context features
    context_features, context_names = extract_features_batch(tweet_texts)

    # 3. Confidence gap features (max_prob - second_max_prob per model)
    gap_features = compute_confidence_gaps(model_probs_list)  # (N, M)
    gap_names = [f"{name}_conf_gap" for name in model_names]

    # 4. Style one-hot features (if provided)
    if style_labels is not None:
        from model_characterisation import style_to_onehot, STYLE_CATEGORIES
        style_onehot = style_to_onehot(style_labels)  # (N, 4)
        style_names = [f"style_{s}" for s in STYLE_CATEGORIES]
    else:
        style_onehot = np.zeros((len(tweet_texts), 0), dtype=np.float32)
        style_names = []

    # Concatenate all features
    meta_features = np.concatenate(
        [prob_features, context_features, gap_features, style_onehot],
        axis=1,
    )
    feature_names = prob_names + context_names + gap_names + style_names

    return meta_features, feature_names


def train_meta_learner(
    val_model_probs_list,
    val_labels,
    val_tweet_texts,
    val_style_labels=None,
    meta_learner_type="mlp",
):
    """
    Train the meta-learner on validation data.

    The meta-learner learns to predict the correct class given
    the softmax outputs of all models + context features + style features
    + confidence gap features.
    At inference, the meta-learner's class probabilities implicitly
    weight each model's contribution based on the input context.

    Args:
        val_model_probs_list: list of M arrays (N_val, C) of softmax probs
        val_labels: array of shape (N_val,) true labels
        val_tweet_texts: list of N_val tweet strings
        val_style_labels: optional list of style labels for validation tweets
        meta_learner_type: 'lr' for logistic regression, 'mlp' for MLP

    Returns:
        meta_learner: trained sklearn model
        scaler: fitted StandardScaler
        feature_names: list of feature names
    """
    meta_features, feature_names = build_meta_features(
        val_model_probs_list, val_tweet_texts, val_style_labels
    )

    # Standardize features
    scaler = StandardScaler()
    meta_features_scaled = scaler.fit_transform(meta_features)

    # Train meta-learner
    if meta_learner_type == "lr":
        meta_learner = LogisticRegression(
            max_iter=1000, multi_class="multinomial", solver="lbfgs",
            random_state=42,
        )
    elif meta_learner_type == "mlp":
        meta_learner = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
        )
    else:
        raise ValueError(f"Unknown meta-learner type: {meta_learner_type}")

    meta_learner.fit(meta_features_scaled, val_labels)

    # Report validation performance
    val_preds = meta_learner.predict(meta_features_scaled)
    val_f1 = f1_score(val_labels, val_preds, average="macro")
    val_acc = accuracy_score(val_labels, val_preds)
    print(f"  Meta-learner validation performance:")
    print(f"    Macro F1:  {val_f1:.4f}")
    print(f"    Accuracy:  {val_acc:.4f}")

    return meta_learner, scaler, feature_names


def predict_with_dynamic_ensemble(
    meta_learner,
    scaler,
    test_model_probs_list,
    test_tweet_texts,
    test_style_labels=None,
):
    """
    Make predictions using the dynamic ensemble.

    Args:
        meta_learner: trained meta-learner
        scaler: fitted scaler
        test_model_probs_list: list of M arrays (N_test, C)
        test_tweet_texts: list of N_test tweet strings
        test_style_labels: optional list of style labels

    Returns:
        ensemble_probs: (N_test, C) ensemble probability matrix
        ensemble_preds: (N_test,) predicted class indices
    """
    meta_features, _ = build_meta_features(
        test_model_probs_list, test_tweet_texts, test_style_labels
    )
    meta_features_scaled = scaler.transform(meta_features)

    # Get meta-learner probabilities (these ARE the ensemble predictions)
    ensemble_probs = meta_learner.predict_proba(meta_features_scaled)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)

    return ensemble_probs, ensemble_preds


def evaluate_dynamic_ensemble(
    ensemble_preds,
    ensemble_probs,
    true_labels,
    class_names,
):
    """Evaluate and report the dynamic ensemble performance."""
    macro_f1 = f1_score(true_labels, ensemble_preds, average="macro")
    acc = accuracy_score(true_labels, ensemble_preds)

    print(f"\nDynamic Ensemble Results ({len(MODEL_NAMES)} models):")
    print(f"  Macro F1 : {macro_f1:.4f}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(
        true_labels, ensemble_preds, target_names=class_names
    ))

    return {"macro_f1": macro_f1, "accuracy": acc}


def save_ensemble_artifacts(
    meta_learner, scaler, ensemble_probs, ensemble_preds,
    true_labels, output_dir,
):
    """Save all ensemble artifacts."""
    os.makedirs(output_dir, exist_ok=True)

    # Save meta-learner and scaler
    joblib.dump(meta_learner, os.path.join(output_dir, "meta_learner.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

    # Save predictions
    np.savez(
        os.path.join(output_dir, "ensemble_predictions.npz"),
        probs=ensemble_probs,
        preds=ensemble_preds,
        labels=true_labels,
    )

    print(f"  Ensemble artifacts saved to {output_dir}")

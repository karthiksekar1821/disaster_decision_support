"""
Explainability-Driven Model Characterisation (Addition 4).

Step 1: Tweet Style Classification (URGENT, FORMAL, EYEWITNESS, INFORMATIONAL)
Step 2: Per-Model Per-Style Performance Matrix (6×4 heatmap)
Step 3: Attribution-Based Style Verification
Step 4: Features for dynamic ensemble (done in dynamic_ensemble.py)

Usage (from Kaggle/Colab):
    from model_characterisation import (
        classify_tweet_style, classify_tweets_batch,
        compute_style_performance_matrix, run_attribution_style_verification,
    )
"""

import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

from disaster_vocab import DISASTER_VOCAB, IRRELEVANT_TOKENS


# ── Tweet Style Categories ───────────────────────────────────────────────────

STYLE_CATEGORIES = ["URGENT", "FORMAL", "EYEWITNESS", "INFORMATIONAL"]

# Style-specific vocabulary and patterns
URGENT_KEYWORDS = {
    "sos", "help", "trapped", "emergency", "urgent", "asap", "rescue",
    "dying", "critical", "immediate", "needed", "please", "save", "hurry",
}

FORMAL_KEYWORDS = {
    "according", "reported", "officials", "authorities", "government",
    "president", "minister", "department", "organization", "agency",
    "statement", "confirmed", "announced", "declared", "assessment",
    "estimated", "approximately", "percent", "million", "billion",
    "fema", "un", "unicef", "who", "ngo", "red cross", "redcross",
}

EYEWITNESS_MARKERS = {
    "i see", "i saw", "i can see", "i hear", "i heard", "i am",
    "i'm", "we are", "we're", "i feel", "i felt", "right now",
    "just happened", "happening now", "in front of", "near me",
    "my house", "my area", "my neighborhood", "my street",
    "i'm at", "we need", "i need", "looking at", "watching",
}

EYEWITNESS_PRONOUNS = {"i", "me", "my", "myself", "we", "our", "us"}

INFORMATIONAL_KEYWORDS = {
    "update", "report", "reports", "according", "sources", "confirmed",
    "forecast", "expected", "category", "magnitude", "level", "warning",
    "advisory", "watch", "alert", "status", "currently", "ongoing",
    "aftermath", "recovery", "damage assessment", "statistics",
    "affected", "impacted", "total", "estimated", "previous",
}


def _score_urgent(text, words):
    """Score how well a tweet matches the URGENT style."""
    score = 0.0
    text_lower = text.lower()

    # All-caps words (strong signal)
    caps_words = [w for w in text.split() if w.isupper() and len(w) > 1]
    score += len(caps_words) * 2.0

    # Exclamation marks
    score += text.count("!") * 1.5

    # Urgent keywords
    for w in words:
        if w in URGENT_KEYWORDS:
            score += 3.0

    # SOS/HELP patterns
    if re.search(r'\bSOS\b', text):
        score += 5.0
    if re.search(r'\bHELP\b', text):
        score += 4.0
    if re.search(r'\bTRAPPED\b', text):
        score += 4.0
    if re.search(r'\bEMERGENCY\b', text):
        score += 4.0

    return score


def _score_formal(text, words):
    """Score how well a tweet matches the FORMAL style."""
    score = 0.0
    text_lower = text.lower()

    # Organisation names and formal keywords
    for w in words:
        if w in FORMAL_KEYWORDS:
            score += 3.0

    # Percentages
    if re.search(r'\d+%', text) or "percent" in text_lower:
        score += 3.0

    # Numbers with commas (structured data)
    if re.search(r'\d{1,3}(,\d{3})+', text):
        score += 2.0

    # Third person reporting (he/she/they said, officials said)
    if re.search(r'(officials?|authorities|government)\s+(said|reported|confirmed)', text_lower):
        score += 4.0

    # Structured reporting language
    if re.search(r'(according to|as per|based on|in a statement)', text_lower):
        score += 3.0

    return score


def _score_eyewitness(text, words):
    """Score how well a tweet matches the EYEWITNESS style."""
    score = 0.0
    text_lower = text.lower()

    # First person pronouns
    first_person_count = sum(1 for w in words if w in EYEWITNESS_PRONOUNS)
    score += first_person_count * 2.0

    # Eyewitness marker phrases
    for marker in EYEWITNESS_MARKERS:
        if marker in text_lower:
            score += 4.0

    # Present tense indicators
    if re.search(r'\b(right now|happening|currently|just)\b', text_lower):
        score += 2.0

    # Location markers
    if re.search(r'\b(here|nearby|close to|near|around)\b', text_lower):
        score += 1.5

    return score


def _score_informational(text, words):
    """Score how well a tweet matches the INFORMATIONAL style."""
    score = 0.0
    text_lower = text.lower()

    # Informational keywords
    for w in words:
        if w in INFORMATIONAL_KEYWORDS:
            score += 2.5

    # Past tense indicators (factual reporting)
    if re.search(r'\b(was|were|had|has been|have been|reported|confirmed)\b', text_lower):
        score += 1.5

    # Statistics and numbers
    numbers = re.findall(r'\d+', text)
    if len(numbers) >= 2:
        score += 2.0

    # Third person neutral
    if re.search(r'\b(the|those|people|residents|community)\b', text_lower):
        score += 1.0

    return score


def classify_tweet_style(text):
    """
    Classify a single tweet into one of 4 style categories.

    Returns the style with the highest match score.
    If no patterns match, defaults to INFORMATIONAL.
    """
    words = set(text.lower().split())

    scores = {
        "URGENT": _score_urgent(text, words),
        "FORMAL": _score_formal(text, words),
        "EYEWITNESS": _score_eyewitness(text, words),
        "INFORMATIONAL": _score_informational(text, words),
    }

    best_style = max(scores, key=scores.get)

    # Default to INFORMATIONAL if all scores are 0
    if scores[best_style] == 0:
        best_style = "INFORMATIONAL"

    return best_style, scores


def classify_tweets_batch(texts):
    """
    Classify a batch of tweets into style categories.

    Returns:
        style_labels: list of style label strings
        style_scores: list of score dicts
    """
    style_labels = []
    style_scores = []
    for text in texts:
        style, scores = classify_tweet_style(text)
        style_labels.append(style)
        style_scores.append(scores)
    return style_labels, style_scores


def style_to_onehot(style_labels):
    """Convert style labels to one-hot encoded array."""
    style_to_idx = {s: i for i, s in enumerate(STYLE_CATEGORIES)}
    n = len(style_labels)
    onehot = np.zeros((n, len(STYLE_CATEGORIES)), dtype=np.float32)
    for i, label in enumerate(style_labels):
        idx = style_to_idx.get(label, 3)  # default to INFORMATIONAL
        onehot[i, idx] = 1.0
    return onehot


# ── Step 2: Per-Model Per-Style Performance Matrix ───────────────────────────

def compute_style_performance_matrix(
    model_predictions,
    style_labels,
    model_names,
    save_dir=None,
):
    """
    Compute Macro F1 per model broken down by tweet style.

    Args:
        model_predictions: dict of model_name -> {"preds": array, "labels": array}
        style_labels: list of style strings for each sample
        model_names: list of model name strings
        save_dir: directory to save results (optional)

    Returns:
        performance_matrix: dict of {model_name: {style: macro_f1}}
    """
    style_labels_arr = np.array(style_labels)
    performance_matrix = {}

    for model_name in model_names:
        preds = model_predictions[model_name]["preds"]
        labels = model_predictions[model_name]["labels"]
        performance_matrix[model_name] = {}

        for style in STYLE_CATEGORIES:
            mask = style_labels_arr == style
            if mask.sum() == 0:
                performance_matrix[model_name][style] = 0.0
                continue

            style_preds = preds[mask]
            style_labels_subset = labels[mask]

            try:
                f1 = f1_score(style_labels_subset, style_preds, average="macro", zero_division=0.0)
            except Exception:
                f1 = 0.0

            performance_matrix[model_name][style] = float(f1)

    # Save to JSON
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        json_path = os.path.join(save_dir, "model_style_performance.json")
        with open(json_path, "w") as f:
            json.dump(performance_matrix, f, indent=2)
        print(f"  Style performance matrix saved to {json_path}")

    return performance_matrix


def plot_style_performance_heatmap(
    performance_matrix,
    model_names,
    save_path=None,
):
    """Generate a heatmap of model × style performance."""
    # Build matrix
    matrix = np.zeros((len(model_names), len(STYLE_CATEGORIES)))
    for i, model in enumerate(model_names):
        for j, style in enumerate(STYLE_CATEGORIES):
            matrix[i, j] = performance_matrix.get(model, {}).get(style, 0.0)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix, annot=True, fmt=".3f", cmap="YlOrRd",
        xticklabels=STYLE_CATEGORIES,
        yticklabels=model_names,
        linewidths=0.5, ax=ax, vmin=0, vmax=1,
    )
    ax.set_title("Model × Tweet Style Performance (Macro F1)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Tweet Style", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Style heatmap saved to {save_path}")
    plt.close()


# ── Step 3: Attribution-Based Style Verification ─────────────────────────────

def _get_style_vocab():
    """Get vocabulary sets for each style category."""
    return {
        "urgent": URGENT_KEYWORDS,
        "formal": FORMAL_KEYWORDS,
        "eyewitness": EYEWITNESS_PRONOUNS | set(
            word for phrase in EYEWITNESS_MARKERS for word in phrase.split()
        ),
        "stopwords": {t for t in IRRELEVANT_TOKENS if len(t) > 1 and not t.startswith("[")},
    }


def compute_attribution_profiles(
    model,
    tokenizer,
    texts,
    style_labels,
    model_name,
    device="cuda",
    n_samples_per_style=30,
    n_steps=50,
):
    """
    Compute average attribution scores for style-specific vocabulary.

    Args:
        model: transformer model
        tokenizer: tokenizer
        texts: list of tweet texts
        style_labels: list of style labels for each text
        model_name: name of the model
        device: 'cuda' or 'cpu'
        n_samples_per_style: max samples per style to analyze
        n_steps: IG steps

    Returns:
        profile: dict of {style: {vocab_category: avg_attribution}}
    """
    from attribution_filter import compute_attributions_for_batch

    style_vocab = _get_style_vocab()
    texts_arr = np.array(texts)
    styles_arr = np.array(style_labels)
    profile = {}

    for style in STYLE_CATEGORIES:
        mask = styles_arr == style
        style_indices = np.where(mask)[0]

        if len(style_indices) == 0:
            profile[style] = {cat: 0.0 for cat in style_vocab}
            continue

        # Sample up to n_samples_per_style
        if len(style_indices) > n_samples_per_style:
            rng = np.random.RandomState(42)
            style_indices = rng.choice(style_indices, n_samples_per_style, replace=False)

        sample_texts = texts_arr[style_indices].tolist()

        # Get model predictions for these samples
        model.eval()
        model.to(device)
        preds = []
        for text in sample_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                             padding="max_length", max_length=128).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
                preds.append(logits.argmax(dim=-1).item())

        # Compute attributions
        try:
            attr_results = compute_attributions_for_batch(
                model=model, tokenizer=tokenizer,
                texts=sample_texts,
                predicted_classes=np.array(preds),
                device=device, n_steps=n_steps,
            )
        except Exception as e:
            print(f"  Warning: attribution computation failed for {model_name}/{style}: {e}")
            profile[style] = {cat: 0.0 for cat in style_vocab}
            continue

        # Compute average attribution per vocabulary category
        cat_scores = {cat: [] for cat in style_vocab}

        for result in attr_results:
            tokens = result["tokens"]
            attributions = np.abs(result["attributions"])

            for token, attr_val in zip(tokens, attributions):
                clean = token.strip("▁").strip("Ġ").strip("##").lower()
                for cat, vocab in style_vocab.items():
                    if clean in vocab:
                        cat_scores[cat].append(float(attr_val))

        profile[style] = {
            cat: float(np.mean(scores)) if scores else 0.0
            for cat, scores in cat_scores.items()
        }

    return profile


def run_attribution_style_verification(
    models_dict,
    tokenizers_dict,
    texts,
    style_labels,
    device="cuda",
    n_samples_per_style=30,
    save_dir=None,
):
    """
    Run attribution-based style verification for all models.

    Args:
        models_dict: dict of model_name -> model
        tokenizers_dict: dict of model_name -> tokenizer
        texts: list of tweet texts
        style_labels: list of style labels
        device: 'cuda' or 'cpu'
        n_samples_per_style: samples per style
        save_dir: directory to save results

    Returns:
        all_profiles: dict of {model_name: {style: {vocab_cat: score}}}
    """
    import torch

    all_profiles = {}

    for model_name in models_dict:
        print(f"  Computing attribution profile for {model_name}...")
        profile = compute_attribution_profiles(
            model=models_dict[model_name],
            tokenizer=tokenizers_dict[model_name],
            texts=texts,
            style_labels=style_labels,
            model_name=model_name,
            device=device,
            n_samples_per_style=n_samples_per_style,
        )
        all_profiles[model_name] = profile

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        json_path = os.path.join(save_dir, "model_attribution_profiles.json")
        with open(json_path, "w") as f:
            json.dump(all_profiles, f, indent=2)
        print(f"  Attribution profiles saved to {json_path}")

    return all_profiles


def plot_attribution_profiles(all_profiles, save_path=None):
    """
    Plot attribution profiles per model per style.

    Creates a grouped bar chart showing average attribution for each
    vocabulary category across models and styles.
    """
    model_names = list(all_profiles.keys())
    vocab_cats = ["urgent", "formal", "eyewitness", "stopwords"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Attribution Profiles per Model per Tweet Style",
                 fontsize=16, fontweight="bold")

    for idx, style in enumerate(STYLE_CATEGORIES):
        ax = axes[idx // 2][idx % 2]
        x = np.arange(len(vocab_cats))
        width = 0.8 / max(len(model_names), 1)

        for i, model in enumerate(model_names):
            vals = [
                all_profiles.get(model, {}).get(style, {}).get(cat, 0.0)
                for cat in vocab_cats
            ]
            ax.bar(x + i * width, vals, width, label=model, alpha=0.85)

        ax.set_title(f"Style: {style}", fontsize=13, fontweight="bold")
        ax.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax.set_xticklabels(vocab_cats, fontsize=10)
        ax.set_ylabel("Avg Attribution", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Attribution profiles plot saved to {save_path}")
    plt.close()

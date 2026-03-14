"""
Context Feature Extraction for Dynamic Ensemble (Novelty 1).

Extracts per-tweet linguistic and contextual features used as input
to the meta-learner that produces per-sample ensemble weights.
"""

import re
import numpy as np


# ── Disaster Keywords ────────────────────────────────────────────────────────

DISASTER_KEYWORDS = {
    "earthquake", "flood", "hurricane", "tornado", "tsunami", "wildfire",
    "fire", "storm", "cyclone", "typhoon", "landslide", "avalanche",
    "collapse", "damage", "destroy", "devastation", "emergency", "crisis",
    "disaster", "catastrophe", "casualty", "death", "rescue", "relief",
    "evacuate", "evacuation", "shelter", "trapped", "missing", "survivor",
    "victim", "injured", "killed", "help", "donate", "volunteer", "aid",
    "supply", "water", "food", "power", "electricity", "bridge", "road",
    "building", "hospital", "school", "house", "home", "displacement",
}

# Urgency markers
URGENCY_KEYWORDS = {
    "urgent", "emergency", "critical", "immediate", "asap", "now",
    "help", "needed", "please", "sos", "trapped", "dying",
}


def extract_features(tweet_text):
    """
    Extract context features from a single tweet.

    Returns a dict of features.
    """
    text = tweet_text.strip()
    words = text.split()
    word_count = len(words)

    features = {
        # Length features
        "char_count": len(text),
        "word_count": word_count,
        "avg_word_length": (
            np.mean([len(w) for w in words]) if word_count > 0 else 0.0
        ),
        # Punctuation features
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        # Case features
        "uppercase_ratio": (
            sum(1 for c in text if c.isupper()) / max(len(text), 1)
        ),
        # Numeric features
        "has_numbers": float(bool(re.search(r'\d', text))),
        # Disaster keyword overlap
        "disaster_keyword_count": sum(
            1 for w in words if w.lower() in DISASTER_KEYWORDS
        ),
        "disaster_keyword_ratio": (
            sum(1 for w in words if w.lower() in DISASTER_KEYWORDS)
            / max(word_count, 1)
        ),
        # Urgency score
        "urgency_keyword_count": sum(
            1 for w in words if w.lower() in URGENCY_KEYWORDS
        ),
    }
    return features


def extract_features_batch(tweet_texts):
    """
    Extract context features for a list of tweets.

    Returns a numpy array of shape (N, num_features).
    """
    all_features = [extract_features(t) for t in tweet_texts]
    feature_names = sorted(all_features[0].keys())
    matrix = np.array([
        [f[name] for name in feature_names]
        for f in all_features
    ], dtype=np.float32)
    return matrix, feature_names

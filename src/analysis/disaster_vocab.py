"""
Disaster-Relevant Vocabulary (Novelty 3 support).

Curated disaster-relevant terms organized by class, used to verify
that model attributions focus on meaningful tokens.
"""

# ── Per-Class Disaster Vocabulary ────────────────────────────────────────────
# These are tokens that SHOULD appear in the top attributions if the
# model's prediction is correct and meaningful.

DISASTER_VOCAB = {
    "infrastructure_and_utility_damage": {
        "damage", "damaged", "destroy", "destroyed", "collapse", "collapsed",
        "bridge", "building", "road", "highway", "infrastructure", "utility",
        "power", "electricity", "outage", "blackout", "water", "gas",
        "flood", "flooded", "flooding", "earthquake", "shattered", "broken",
        "rubble", "debris", "wreck", "wreckage", "crack", "cracked",
        "structure", "foundation", "tower", "pipe", "pipeline", "levee",
        "dam", "roof", "wall", "window", "school", "hospital", "church",
        "house", "home", "apartment", "devastation", "devastated",
        "inundated", "submerged", "underwater", "erosion",
    },
    "rescue_volunteering_or_donation_effort": {
        "rescue", "rescued", "rescuer", "volunteer", "volunteering",
        "donate", "donation", "donations", "relief", "aid", "help",
        "helping", "shelter", "evacuate", "evacuation", "supply",
        "supplies", "fund", "funds", "fundraise", "fundraising",
        "charity", "humanitarian", "organization", "ngo", "redcross",
        "red cross", "fema", "support", "supporting", "contribute",
        "contribution", "effort", "efforts", "team", "search",
        "blanket", "tent", "medicine", "medical", "doctor", "nurse",
        "food", "clothing", "water", "distribute", "distribution",
    },
    "injured_or_dead_people": {
        "dead", "death", "deaths", "died", "killed", "kill",
        "injured", "injury", "injuries", "hurt", "wound", "wounded",
        "body", "bodies", "victim", "victims",
        "casualty", "casualties", "fatality", "fatalities", "toll",
        "confirmed", "unconfirmed", "number", "count",
    },
    "displaced_people_and_evacuations": {
        "displaced", "evacuated", "evacuation", "evacuee", "evacuees",
        "refugee", "refugees", "homeless", "shelter", "shelters",
        "camp", "camps", "relocation", "relocated", "flee", "fleeing",
        "stranded", "family", "families", "people", "person",
    },
    "missing_or_found_people": {
        "missing", "found", "lost", "search", "searching",
        "trapped", "survivor", "survivors", "alive", "rescue",
        "child", "children", "baby", "elderly", "person", "people",
        "locate", "located", "unaccounted",
    },
    "other_relevant_information": {
        "update", "report", "reports", "news",
        "situation", "information", "info", "status", "condition",
        "latest", "current", "ongoing", "continue", "continues",
        "category", "forecast", "path", "track", "approaching",
        "landfall", "aftermath", "recovery", "impact", "affected",
        "area", "region", "zone", "state", "country", "city",
    },
    "caution_and_advice": {
        "warning", "alert", "caution", "advisory", "advice",
        "safety", "safe", "avoid", "prepare", "preparation",
        "evacuate", "evacuation", "order", "mandatory", "voluntary",
        "shelter", "stay", "leave", "danger", "dangerous",
        "emergency", "declared", "declaration",
        "official", "government", "authority", "president", "governor",
        "mayor",
    },
    "requests_or_urgent_needs": {
        "need", "needed", "needs", "urgent", "urgently",
        "request", "requesting", "require", "required",
        "help", "please", "sos", "supply", "supplies",
        "food", "water", "medicine", "medical", "blanket",
        "shelter", "rescue", "immediate", "critical", "essential",
    },
    "sympathy_and_support": {
        "pray", "praying", "prayers", "thought", "thoughts",
        "heart", "hearts", "condolence", "condolences",
        "sympathy", "support", "solidarity", "love",
        "hope", "hoping", "stay strong", "god", "bless",
        "rip", "rest", "peace", "sad", "tragic", "tragedy",
        "sorry", "devastated", "heartbroken",
    },
    "not_humanitarian": set(),  # No specific disaster keywords expected
}


# ── Stopwords / Irrelevant Tokens ────────────────────────────────────────────
# Tokens that should NOT dominate attributions for correct predictions

IRRELEVANT_TOKENS = {
    # Common English stopwords
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "and", "but", "if", "or", "because", "as", "until", "while",
    "of", "at", "by", "for", "with", "about", "against", "between",
    "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "just", "don", "now",
    # Social media artifacts (post-cleaning)
    "rt", "via", "lol", "omg", "smh", "tbh", "imo", "fwiw",
    # Punctuation tokens (from tokenizers)
    ".", ",", "!", "?", ":", ";", "-", "–", "—", "'", '"',
    "(", ")", "[", "]", "{", "}", "/", "\\", "|", "@", "#",
    # Common padding / special tokens
    "[PAD]", "[CLS]", "[SEP]", "[UNK]", "<s>", "</s>", "<pad>",
}


def get_disaster_vocab_for_class(class_name):
    """Get the disaster vocabulary for a specific class."""
    return DISASTER_VOCAB.get(class_name, set())


def is_disaster_relevant(token, class_name):
    """Check if a token is disaster-relevant for the given class."""
    vocab = get_disaster_vocab_for_class(class_name)
    return token.lower().strip() in vocab


def is_irrelevant(token):
    """Check if a token is an irrelevant/stopword token."""
    return token.lower().strip() in IRRELEVANT_TOKENS

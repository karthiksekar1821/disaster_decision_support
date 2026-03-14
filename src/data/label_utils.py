import json

# ── 10-class → 5-class Merge Mapping ────────────────────────────────────────
# Original 10 classes from HumAID → new 5-class schema
CLASS_MERGE_MAP = {
    "infrastructure_and_utility_damage": "infrastructure_and_utility_damage",
    "rescue_volunteering_or_donation_effort": "rescue_volunteering_or_donation_effort",
    "injured_or_dead_people": "affected_individuals",
    "displaced_people_and_evacuations": "affected_individuals",
    "missing_or_found_people": "affected_individuals",
    "other_relevant_information": "other_relevant_information",
    "not_humanitarian": "not_humanitarian",
    # These 3 classes are mapped to other_relevant_information
    "caution_and_advice": "other_relevant_information",
    "requests_or_urgent_needs": "other_relevant_information",
    "sympathy_and_support": "other_relevant_information",
}

# The 5 target classes in canonical order
TARGET_CLASSES = sorted([
    "infrastructure_and_utility_damage",
    "rescue_volunteering_or_donation_effort",
    "affected_individuals",
    "other_relevant_information",
    "not_humanitarian",
])


def merge_classes(dataset_split):
    """Apply the 10→5 class merge to a dataset split."""
    def _merge(example):
        old_label = example["class_label"]
        new_label = CLASS_MERGE_MAP.get(old_label)
        if new_label is None:
            raise ValueError(f"Unknown class label: {old_label}")
        example["class_label"] = new_label
        return example
    return dataset_split.map(_merge)


def create_label_mapping(dataset=None):
    """Create label2id and id2label mappings for the 5-class schema."""
    labels = TARGET_CLASSES
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def encode_labels(split_dataset, label2id):
    def encode(example):
        example["label"] = label2id[example["class_label"]]
        return example
    return split_dataset.map(encode)


def save_label_mapping(label2id, id2label, path="../../data/processed/label_mapping.json"):
    with open(path, "w") as f:
        json.dump(
            {"label2id": label2id, "id2label": id2label},
            f,
            indent=4
        )
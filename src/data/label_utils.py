import json
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight


# ── 5-Class Schema (CrisisMMD) ──────────────────────────────────────────────
# Records with labels NOT in this list are dropped during data preparation.

TARGET_CLASSES = sorted([
    "affected_individuals",
    "infrastructure_and_utility_damage",
    "not_humanitarian",
    "other_relevant_information",
    "rescue_volunteering_or_donation_effort",
])


def filter_to_target_classes(dataset_split):
    """Drop records whose label is not in the 5 target classes."""
    return dataset_split.filter(
        lambda example: example["label"] in TARGET_CLASSES
    )


def create_label_mapping():
    """Create label2id and id2label mappings for the 5-class schema."""
    label2id = {label: idx for idx, label in enumerate(TARGET_CLASSES)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def encode_labels(split_dataset, label2id):
    """Map string label column to integer IDs."""
    def encode(example):
        example["label"] = label2id[example["label"]]
        return example
    return split_dataset.map(encode)


def save_label_mapping(label2id, id2label, path="../../data/processed/label_mapping.json"):
    with open(path, "w") as f:
        json.dump(
            {"label2id": label2id, "id2label": id2label},
            f,
            indent=4,
        )


def compute_and_save_class_weights(labels, num_classes, path="../../data/processed/class_weights.pt"):
    """Compute balanced class weights and save as a PyTorch tensor."""
    labels_array = np.array(labels)
    classes = np.arange(num_classes)
    weights = compute_class_weight("balanced", classes=classes, y=labels_array)
    # Normalize so they sum to num_classes
    weights = weights / weights.sum() * num_classes
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    torch.save(weights_tensor, path)
    print(f"  Class weights saved to {path}")
    print(f"  Weights: {weights_tensor.tolist()}")
    return weights_tensor
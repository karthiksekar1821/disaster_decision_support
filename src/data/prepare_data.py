"""
Data preparation script for CrisisMMD dataset.

Loads raw JSON, cleans text, filters to 5 target classes,
encodes labels, computes class weights, and saves processed parquet.

Run from src/data/:
    python prepare_data.py
"""

from split import load_local_crisismmd, preprocess_split, save_processed_splits
from label_utils import (
    filter_to_target_classes, create_label_mapping,
    encode_labels, save_label_mapping, compute_and_save_class_weights,
    TARGET_CLASSES,
)

NUM_LABELS = len(TARGET_CLASSES)

print("Loading CrisisMMD dataset from JSON...")
dataset = load_local_crisismmd()

print("Preprocessing splits (cleaning text, removing low-info)...")
dataset["train"] = preprocess_split(dataset["train"])
dataset["validation"] = preprocess_split(dataset["validation"])
dataset["test"] = preprocess_split(dataset["test"])

print(f"Filtering to {NUM_LABELS} target classes...")
print(f"  Keeping: {TARGET_CLASSES}")
for split_name in ["train", "validation", "test"]:
    before = len(dataset[split_name])
    dataset[split_name] = filter_to_target_classes(dataset[split_name])
    after = len(dataset[split_name])
    print(f"  {split_name}: {before} → {after} (dropped {before - after})")

print("Creating label mapping (5-class schema)...")
label2id, id2label = create_label_mapping()
print(f"  Labels: {label2id}")

print("Encoding labels...")
dataset["train"] = encode_labels(dataset["train"], label2id)
dataset["validation"] = encode_labels(dataset["validation"], label2id)
dataset["test"] = encode_labels(dataset["test"], label2id)

print("Computing and saving class weights...")
train_labels = dataset["train"]["label"]
compute_and_save_class_weights(train_labels, NUM_LABELS)

print("Saving label mapping...")
save_label_mapping(label2id, id2label)

print("Saving processed splits as parquet...")
save_processed_splits(dataset)

# Print summary statistics
from collections import Counter
for split_name in ["train", "validation", "test"]:
    split = dataset[split_name]
    print(f"\n{split_name}: {len(split)} samples")
    class_counts = Counter(split["label"])
    for cls_id, count in sorted(class_counts.items()):
        print(f"  {id2label[cls_id]}: {count}")

print("\nData preparation complete.")
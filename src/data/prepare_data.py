"""
Data preparation script for HumAID dataset.

Loads raw JSON, cleans text, filters to 5 target classes,
encodes labels, computes class weights, and saves processed parquet.

Run from src/data/:
    python prepare_data.py
"""

from split import load_local_humaid, preprocess_split, save_processed_splits
from label_utils import (
    filter_to_target_classes, create_label_mapping,
    encode_labels, save_label_mapping, compute_and_save_class_weights,
    TARGET_CLASSES,
)

NUM_LABELS = len(TARGET_CLASSES)

print("Loading HumAID dataset from Parquet...")
dataset = load_local_humaid()

print("Preprocessing splits (cleaning text, removing low-info)...")
dataset["train"] = preprocess_split(dataset["train"])
dataset["validation"] = preprocess_split(dataset["validation"])
dataset["test"] = preprocess_split(dataset["test"])

import pandas as pd
from datasets import Dataset

print("Deduplicating splits (majority vote on conflicts, tie-breaker on lower ID)...")
for split_name in ["train", "validation", "test"]:
    before_dedup = len(dataset[split_name])
    df = dataset[split_name].to_pandas()
    
    # Drop exact duplicates (same text, same label)
    df = df.drop_duplicates()
    
    # Resolve conflicts (same text, different label) using majority vote
    # 1. Count occurrences of each (tweet_text, label) pair
    counts = df.groupby(["tweet_text", "label"]).size().reset_index(name="count")
    
    # 2. Sort so that for each tweet, the label with highest count comes first
    #    On tie (same count), alphabetical sort on label breaks tie consistently
    counts = counts.sort_values(by=["tweet_text", "count", "label"], ascending=[True, False, True])
    
    # 3. Drop all duplicates keeping only the first (highest count) label
    resolved = counts.drop_duplicates(subset=["tweet_text"], keep="first")
    
    # 4. Filter the original dataframe to keep only the resolved (text, label) pairs
    merged = pd.merge(df, resolved[["tweet_text", "label"]], on=["tweet_text", "label"], how="inner")
    
    # 5. Drop any lingering duplicates (same text and label) from original df
    merged = merged.drop_duplicates(subset=["tweet_text"])
    
    dataset[split_name] = Dataset.from_pandas(merged)
    # Ensure low_info column remains but any index pandas adds is removed
    if "__index_level_0__" in dataset[split_name].column_names:
        dataset[split_name] = dataset[split_name].remove_columns("__index_level_0__")
        
    print(f"  {split_name}: {before_dedup} → {len(dataset[split_name])} (dropped {before_dedup - len(dataset[split_name])} duplicates)")

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
        print(f"  {id2label[int(cls_id)]}: {count}")

print("\nData preparation complete.")
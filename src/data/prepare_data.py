from split import load_local_humaid, preprocess_split, save_processed_splits
from label_utils import (
    merge_classes, create_label_mapping,
    encode_labels, save_label_mapping
)

print("Loading dataset...")
dataset = load_local_humaid()

print("Preprocessing splits (cleaning text, removing low-info)...")
dataset["train"] = preprocess_split(dataset["train"])
dataset["validation"] = preprocess_split(dataset["validation"])
dataset["test"] = preprocess_split(dataset["test"])

print("Merging classes (10 → 5)...")
dataset["train"] = merge_classes(dataset["train"])
dataset["validation"] = merge_classes(dataset["validation"])
dataset["test"] = merge_classes(dataset["test"])

print("Creating label mapping (5-class schema)...")
label2id, id2label = create_label_mapping()
print(f"  Labels: {label2id}")

print("Encoding labels...")
dataset["train"] = encode_labels(dataset["train"], label2id)
dataset["validation"] = encode_labels(dataset["validation"], label2id)
dataset["test"] = encode_labels(dataset["test"], label2id)

print("Saving label mapping...")
save_label_mapping(label2id, id2label)

print("Saving processed splits...")
save_processed_splits(dataset)

# Print summary statistics
for split_name in ["train", "validation", "test"]:
    split = dataset[split_name]
    print(f"\n{split_name}: {len(split)} samples")
    # Count per class
    from collections import Counter
    class_counts = Counter(split["class_label"])
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count}")

print("\nData preparation complete.")
from split import load_local_humaid, preprocess_split, save_processed_splits
from label_utils import create_label_mapping, encode_labels, save_label_mapping

print("Loading dataset...")
dataset = load_local_humaid()

print("Preprocessing splits...")
dataset["train"] = preprocess_split(dataset["train"])
dataset["validation"] = preprocess_split(dataset["validation"])
dataset["test"] = preprocess_split(dataset["test"])

print("Creating label mapping...")
label2id, id2label = create_label_mapping(dataset)

print("Encoding labels...")
dataset["train"] = encode_labels(dataset["train"], label2id)
dataset["validation"] = encode_labels(dataset["validation"], label2id)
dataset["test"] = encode_labels(dataset["test"], label2id)

print("Saving label mapping...")
save_label_mapping(label2id, id2label)

print("Saving processed splits...")
save_processed_splits(dataset)

print("Data preparation complete.")
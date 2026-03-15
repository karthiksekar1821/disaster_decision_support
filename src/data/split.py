import json
from datasets import Dataset, DatasetDict
from preprocessing import clean_text, is_low_information


def load_local_crisismmd():
    """Load CrisisMMD dataset from raw JSON files."""
    splits = {}
    for split_name, filename in [
        ("train", "../../data/raw/train.json"),
        ("validation", "../../data/raw/val.json"),
        ("test", "../../data/raw/test.json"),
    ]:
        with open(filename) as f:
            records = json.load(f)
        # Keep only tweet_text and label columns
        rows = [{"tweet_text": r["tweet_text"], "label": r["label"]} for r in records]
        splits[split_name] = Dataset.from_list(rows)
    return DatasetDict(splits)


def preprocess_split(split_dataset):
    """Clean text and filter low-information tweets."""
    def process(example):
        cleaned = clean_text(example["tweet_text"])
        example["tweet_text"] = cleaned
        example["low_info"] = is_low_information(cleaned)
        return example

    split_dataset = split_dataset.map(process)
    split_dataset = split_dataset.filter(lambda x: x["tweet_text"].strip() != "")
    return split_dataset


def save_processed_splits(dataset):
    """Save processed splits as parquet files."""
    dataset["train"].to_parquet("../../data/processed/train.parquet")
    dataset["validation"].to_parquet("../../data/processed/val.parquet")
    dataset["test"].to_parquet("../../data/processed/test.parquet")
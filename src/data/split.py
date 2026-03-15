import json
from datasets import Dataset, DatasetDict, load_dataset
from preprocessing import clean_text, is_low_information


def load_local_humaid():
    """Load HumAID dataset from raw Parquet files."""
    dataset = load_dataset("parquet", data_files={
        "train": "../../data/raw/train.parquet",
        "validation": "../../data/raw/val.parquet",
        "test": "../../data/raw/test.parquet",
    })
    
    # Rename 'class_label' to 'label' for pipeline compatibility
    if "class_label" in dataset["train"].column_names:
        dataset = dataset.rename_column("class_label", "label")
        
    # Drop completely irrelevant columns if they exist, but keep tweet_text and label
    def filter_cols(split_ds):
        cols_to_keep = ["tweet_text", "label"]
        cols_to_remove = [c for c in split_ds.column_names if c not in cols_to_keep]
        if cols_to_remove:
            split_ds = split_ds.remove_columns(cols_to_remove)
        return split_ds

    dataset["train"] = filter_cols(dataset["train"])
    dataset["validation"] = filter_cols(dataset["validation"])
    dataset["test"] = filter_cols(dataset["test"])
    
    return dataset


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
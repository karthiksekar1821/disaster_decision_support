from datasets import load_dataset
from preprocessing import clean_text, is_low_information

def load_local_humaid():
    dataset = load_dataset(
        "parquet",
        data_files={
            "train": "../../data/raw/train.parquet",
            "validation": "../../data/raw/val.parquet",
            "test": "../../data/raw/test.parquet",
        },
    )
    return dataset


def preprocess_split(split_dataset):
    def process(example):
        cleaned = clean_text(example["tweet_text"])
        example["tweet_text"] = cleaned
        example["low_info"] = is_low_information(cleaned)
        return example

    split_dataset = split_dataset.map(process)
    split_dataset = split_dataset.filter(lambda x: x["tweet_text"].strip() != "")
    return split_dataset


def save_processed_splits(dataset):
    dataset["train"].to_parquet("../../data/processed/train.parquet")
    dataset["validation"].to_parquet("../../data/processed/val.parquet")
    dataset["test"].to_parquet("../../data/processed/test.parquet")
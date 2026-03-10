from datasets import load_dataset

dataset = load_dataset(
    "parquet",
    data_files={
        "train": "../../data/raw/train.parquet",
        "validation": "../../data/raw/val.parquet",
        "test": "../../data/raw/test.parquet",
    },
)

print(dataset)

print("\nTrain size:", len(dataset["train"]))
print("Validation size:", len(dataset["validation"]))
print("Test size:", len(dataset["test"]))

print("\nSample example:")
print(dataset["train"][0])
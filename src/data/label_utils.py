import json

def create_label_mapping(dataset):
    labels = sorted(list(set(dataset["train"]["class_label"])))
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def encode_labels(split_dataset, label2id):
    def encode(example):
        example["label"] = label2id[example["class_label"]]
        return example

    return split_dataset.map(encode)


def save_label_mapping(label2id, id2label):
    with open("../../data/processed/label_mapping.json", "w") as f:
        json.dump(
            {"label2id": label2id, "id2label": id2label},
            f,
            indent=4
        )
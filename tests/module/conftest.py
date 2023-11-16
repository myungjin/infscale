"""conftest file."""

supported_model_names = [
    # language models
    "bert-base-uncased",
    "gpt2",
    "t5-small",
    # image models
    "google/vit-base-patch16-224",
    "microsoft/resnet-50",
]

datasets = [
    ("wikitext", "wikitext-2-raw-v1", "test"),
    ("tiny_shakespeare", "", "test"),
    ("wikitext", "wikitext-2-raw-v1", "test"),
    ("Maysee/tiny-imagenet", "", "valid"),
    ("cifar10", "", "test"),
]

model_dataset_pairs = list(zip(supported_model_names, datasets))

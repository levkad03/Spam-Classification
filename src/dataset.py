from datasets import load_from_disk
from transformers import AutoTokenizer

from config import Config


def get_tokenized_dataset(config: Config):
    dataset = load_from_disk("spam-detection-dataset")

    label2id = {"not_spam": 0, "spam": 1}
    dataset = dataset.map(lambda x: {"label": label2id[x["label"]]})

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=config.max_length,
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    return dataset

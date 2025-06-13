from dataclasses import dataclass


@dataclass
class Config:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 128
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 2e-5
    output_dir: str = "models/bert_spam_model"

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

import wandb
from config import Config
from dataset import get_tokenized_dataset
from utils import compute_metrics

wandb.init(project="Spam Classification", name="bert-run")


def train():
    config = Config()
    dataset = get_tokenized_dataset(config)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = DistilBertForSequenceClassification.from_pretrained(
        config.model_name, num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        eval_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        save_strategy="epoch",
        logging_dir="./logs",
        load_best_model_at_end=True,
        report_to="wandb",
        logging_strategy="epoch",
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(config.output_dir)


if __name__ == "__main__":
    train()

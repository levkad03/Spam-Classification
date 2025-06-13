from transformers import BertForSequenceClassification, Trainer, TrainingArguments

from config import Config
from dataset import get_tokenized_dataset


def train():
    config = Config()
    dataset = get_tokenized_dataset(config)

    model = BertForSequenceClassification.from_pretrained(
        config.model_name, num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        evaluation_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        save_strategy="epoch",
        logging_dir="./logs",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )

    trainer.train()
    trainer.save_model(config.output_dir)


if __name__ == "__main__":
    train()

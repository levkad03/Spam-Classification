import evaluate

metric_accuracy = evaluate.load("accuracy")
metric_precision = evaluate.load("precision")
metric_recall = evaluate.load("recall")
metric_f1 = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {
        "accuracy": metric_accuracy.compute(predictions=predictions, references=labels)[
            "accuracy"
        ],
        "precision": metric_precision.compute(
            predictions=predictions, references=labels
        )["precision"],
        "recall": metric_recall.compute(predictions=predictions, references=labels)[
            "recall"
        ],
        "f1": metric_f1.compute(predictions=predictions, references=labels)["f1"],
    }

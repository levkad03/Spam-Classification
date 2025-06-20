import logging
from typing import Any, Dict

import torch
from fastapi import FastAPI, HTTPException
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

label_names = ["not_spam", "spam"]


def load_model(app: FastAPI) -> None:
    try:
        model_path = "models/bert_spam_model"
        logger.info(f"Loading model from {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        app.state.model = model
        app.state.tokenizer = tokenizer
        app.state.device = device

        logger.info(f"Model loaded successfully on {device}")

    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise e


def predict_text(app: FastAPI, text: str) -> Dict[str, Any]:
    model = getattr(app.state, "model", None)
    tokenizer = getattr(app.state, "tokenizer", None)
    device = getattr(app.state, "device", None)

    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        inputs = tokenizer(
            text, truncation=True, padding=True, max_length=128, return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()

        probs_dict = {
            label_names[i]: probabilities[0][i].item() for i in range(len(label_names))
        }

        return {
            "text": text,
            "label": label_names[predicted_class],
            "confidence": confidence,
            "probabilities": probs_dict,
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

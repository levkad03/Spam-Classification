import logging
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
tokenizer = None
device = None
label_names = ["not_spam", "spam"]


def load_model():
    global model, tokenizer, device

    try:
        model_path = "models/bert_spam_model"
        logger.info(f"Loading model from {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        logger.info(f"Model loaded successfully on {device}")

    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise e


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application")

    load_model()
    yield

    logger.info("Shutting down application")


app = FastAPI(
    title="Spam Detection API",
    description="REST API for spam detection using BERT model",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Spam Detection API!"}


if __name__ == "__main__":
    uvicorn.run(app, port=8000)

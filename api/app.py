import logging
from contextlib import asynccontextmanager

import uvicorn
from constants import device, label_names, model, tokenizer
from fastapi import FastAPI, HTTPException
from schemas import PredictionResponse, TextInput

from utils import load_model, predict_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


@app.post("/predict")
async def predict_single(input_data: TextInput):
    result = predict_text(input_data.text)

    return PredictionResponse(**result)


@app.get("/model/info")
async def get_model_info():
    global model, tokenizer, device
    if model is None:
        return HTTPException(status_code=500, detail="Model not loaded")

    return {
        "model_type": model.__class__.__name__,
        "device": str(device),
        "labels": label_names,
        "max_length": 128,
        "tokenizer_vocab_size": len(tokenizer) if tokenizer else None,
    }


if __name__ == "__main__":
    uvicorn.run(app, port=8000)

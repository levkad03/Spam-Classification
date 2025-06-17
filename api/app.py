import logging
from contextlib import asynccontextmanager

import uvicorn
from constants import label_names
from fastapi import FastAPI, HTTPException, Request
from schemas import (
    BatchedPredictionResponse,
    BatchTextInput,
    PredictionResponse,
    TextInput,
)

from utils import load_model, predict_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application")

    load_model(app)
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
async def predict_single(request: Request, input_data: TextInput):
    result = predict_text(request.app, input_data.text)

    return PredictionResponse(**result)


@app.post("/predict/batch")
async def predict_batch(request: Request, input_data: BatchTextInput):
    predictions = []

    for text in input_data.texts:
        try:
            result = predict_text(app=request.app, text=text)
            predictions.append(PredictionResponse(**result))
        except Exception as e:
            logger.error(f"Error processing text: {text[:50]}... Error: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Batch prediction failed: {str(e)}"
            )

    return BatchedPredictionResponse(predictions=predictions)


@app.get("/model/info")
async def get_model_info(request: Request):
    model = getattr(request.app.state, "model", None)
    tokenizer = getattr(request.app.state, "tokenizer", None)
    device = getattr(request.app.state, "device", None)

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

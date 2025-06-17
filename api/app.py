import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
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
def predict_single(input_data: TextInput):
    result = predict_text(input_data.text)

    return PredictionResponse(**result)


if __name__ == "__main__":
    uvicorn.run(app, port=8000)

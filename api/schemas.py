from typing import Dict, List

from pydantic import BaseModel, Field


class TextInput(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=5000, description="Text to classify"
    )


class PredictionResponse(BaseModel):
    text: str
    label: str
    confidence: float
    probabilities: Dict[str, float]


class BatchTextInput(BaseModel):
    texts: List[str] = Field(
        ..., min_items=1, max_items=100, description="List of texts to classify"
    )


class BatchedPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

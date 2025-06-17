from typing import Dict

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

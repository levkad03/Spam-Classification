import uvicorn
from fastapi import FastAPI

app = FastAPI(
    title="Spam Detection API",
    description="REST API for spam detection using BERT model",
    version="1.0.0",
)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Spam Detection API!"}


if __name__ == "__main__":
    uvicorn.run(app, port=8000)

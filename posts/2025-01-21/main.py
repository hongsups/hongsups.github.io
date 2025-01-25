from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List
import uvicorn

# 1. Initialize the FastAPI app
app = FastAPI(
    title="Model API",
    description="API to make prediction_scores with a trained model",
    version="0.1",
)


# 2. Define input and out data model using pydantic
class InputData(BaseModel):
    features: List[float]


class OutputData(BaseModel):
    prediction_score: float


# 3. Load the model
model = joblib.load("model.joblib")


@app.get("/")
def home():
    """Health check endpoint"""
    return {"message": "ML model API is running", "status": "healthy"}


@app.post("/predict", response_model=OutputData)
def predict(data: InputData):
    """Make prediction scores with the trained model"""
    try:
        features = np.array(data.features).reshape(1, -1)  # sklearn expects 2D array
        prediction_score = float(model.predict_proba(features)[0][1])
        return OutputData(prediction_score=prediction_score)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

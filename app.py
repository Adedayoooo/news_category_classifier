from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from classifier import predict

app = FastAPI(
    title="News Classification API",
    description="BERT-based news topic classification",
    version="1.0.0"
)
class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=512)
class PredictionResponse(BaseModel):
    text: str
    label: str
    confidence: float
    
@app.get("/")
def root():
    return {"status": "News Classification API is running"}

@app.post("/predict", response_model=PredictionResponse)
def classify(data: TextInput):
    try:
        result = predict(data.text)
        return {
            "text": data.text,
            "label": result["label"],
            "confidence": round(result["confidence"], 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
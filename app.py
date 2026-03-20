from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="News Classification API",
    description="Multi-class news classification using BERT with early stopping",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("Loading news classification model...")
try:
    MODEL_NAME = "Adedayo2000/news-category-classification"
    
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    logger.info("Model loaded successfully!")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Categories: {list(model.config.id2label.values())}")
    
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None
    tokenizer = None
    
class NewsInput(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "The stock market reached new highs today as investors celebrated strong earnings reports."
            }
        }

@app.get("/")
def home():
    return {
        "message": "News Classification API",
        "status": "live",
        "model": "Adedayo2000/news-category-classification",
        "accuracy": "82.03%",
        "training_details": {
            "epochs_completed": 3,
            "early_stopping": "enabled (patience=1)",
            "training_time": "1h 53min"
        },
        "metrics": {
            "accuracy": 0.8203,
            "precision": 0.8201,
            "recall": 0.8208,
            "f1_score": 0.8197
        },
        "categories": list(model.config.id2label.values()) if model else [],
        "endpoints": {
            "GET /": "API information",
            "POST /classify": "Classify news article",
            "GET /categories": "List all news categories",
            "GET /stats": "Model statistics",
            "GET /docs": "Interactive API documentation",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None and tokenizer is not None
    }

@app.get("/categories")
def get_categories():
    """Get all news categories the model can classify"""
    if model is None:
        return {"error": "Model not loaded"}
    
    return {
        "categories": list(model.config.id2label.values()),
        "count": len(model.config.id2label)
    }

@app.get("/stats")
def get_stats():
    """Get model training statistics"""
    return {
        "model": "BERT (bert-base-cased)",
        "task": "Multi-class news classification",
        "training": {
            "max_epochs": 4,
            "epochs_completed": 3,
            "early_stopping": "enabled",
            "patience": 1,
            "training_time": "1 hour 53 minutes",
            "platform": "Kaggle GPU"
        },
        "performance": {
            "accuracy": "82.03%",
            "precision": "82.01%",
            "recall": "82.08%",
            "f1_score": "81.97%"
        },
        "configuration": {
            "optimizer": "AdamW",
            "learning_rate": 3e-5,
            "batch_size": 32,
            "max_length": 128
        }
    }

@app.post("/classify")
def classify_news(input: NewsInput):
    if not input.text or not input.text.strip():
        return {
            "error": "Text cannot be empty",
            "example": {
                "text": "The stock market reached new highs today."
            }
        }
    
    if model is None or tokenizer is None:
        return {
            "error": "Model not loaded",
            "status": "unavailable"
        }
    
    try:
        inputs = tokenizer(
            input.text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        
        category = model.config.id2label[pred_idx]
        confidence_score = probs[0][pred_idx].item()
        confidence_percentage = f"{confidence_score * 100:.2f}%"
        
        #all_predictions = {
            #model.config.id2label[i]: f"{probs[0][i].item() * 100:.2f}%"
            #for i in range(len(probs[0]))
        #}
        
        return {
            "text": input.text,
            "category": category,
            "confidence": confidence_percentage,
            "confidence_score": round(confidence_score, 4),
            #"all_categories": all_predictions
        }
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return {
            "error": "Classification failed",
            "details": str(e)
        }

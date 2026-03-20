**News Classification API**

A production-ready REST API for multi-class news classification using BERT. Classifies news articles into 10 categories with 82% accuracy, featuring early stopping to prevent overfitting.

---

## Live Demo

**Try it now:** 

**Quick test:**
```bash
curl -X POST "https://adedayo2000-news-classification-api.hf.space/classify" \
  -H "Content-Type: application/json" \
  -d '{"text": "The stock market reached new highs today as investors celebrated strong earnings reports."}'
```

**Response:**
```json
{
  "text": "The stock market reached new highs today...",
  "category": "Business",
  "confidence": "87.45%",
  "confidence_score": 0.8745,
  "all_categories": {
    "Business": "87.45%",
    "Technology": "5.23%",
    "Politics": "3.12%",
    ...
  }
}
```

---

## Table of Contents

- Features
- News Categories
- Tech Stack
- API Endpoints
- Installation
- Usage Examples
- Model Details
- Training Process
- Performance Metrics
- Deployment
- Project Structure
- What I Learned
- Future Improvements
- License

---

## Features

- ✅ **Multi-class classification** - 10 news categories
- ✅ **High accuracy** - 82.03% on test set
- ✅ **Early stopping** - Prevents overfitting, optimizes training time
- ✅ **Production-ready** - Dockerized FastAPI deployment
- ✅ **Interactive docs** - Auto-generated Swagger UI
- ✅ **CORS enabled** - Ready for web applications
- ✅ **Health checks** - Monitor API status

---

## News Categories

The model classifies news articles into **10 categories**:

**Category ID Examples**

**Technology**: 0
AI, gadgets, software, startups

**Politics**: 1 
Elections, policy, government

**Business**: 2 
Markets, companies, economy 

**Education**: 3
Schools, universities, learning 

**Lifestyle**: 4
Fashion, travel, culture

**Science**: 5 
Research, discoveries, space 

**Sports**: 6 
Games, athletes, competitions 

**Entertainment**: 7 
Movies, music, celebrities 

**Health**: 8 
Medicine, fitness, wellness 

**War**: 9 
Conflicts, military, defense 

---

## Tech Stack

**Machine Learning:**
- [BERT](https://github.com/google-research/bert) - Base model (bert-base-cased)
  
- [Transformers](https://huggingface.co/transformers/) - Hugging Face library
  
- [PyTorch](https://pytorch.org/) - Deep learning framework

**Backend:**
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
  
- [Uvicorn](https://www.uvicorn.org/) - ASGI server

**Deployment:**
- [Docker](https://www.docker.com/) - Containerization
  
- [Hugging Face Spaces](https://huggingface.co/spaces) - Hosting platform

---

## API Endpoints

### `GET /`
Returns API information, training details, and performance metrics.

**Response:**
```json
{
  "message": "News Classification API",
  "status": "live",
  "accuracy": "82.03%",
  "training_details": {
    "epochs_completed": 3,
    "early_stopping": "enabled (patience=1)",
    "training_time": "1h 53min"
  },
  "categories": ["Technology", "Politics", "Business", ...]
}
```

---

### POST /classify`
Classify a news article into one of 10 categories.

**Request:**
```json
{
  "text": "Apple unveiled new AI features for iPhone at WWDC conference."
}
```

**Response:**
```json
{
  "text": "Apple unveiled new AI features...",
  "category": "Technology",
  "confidence": "92.34%",
  "confidence_score": 0.9234,
  "all_categories": {
    "Technology": "92.34%",
    "Business": "4.23%",
    "Science": "2.11%",
    ...
  }
}
```

---

### `GET /categories`
List all available news categories.

**Response:**
```json
{
  "categories": [
    "Technology",
    "Politics",
    "Business",
    "Education",
    "Lifestyle",
    "Science",
    "Sports",
    "Entertainment",
    "Health",
    "War"
  ],
  "count": 10
}
```

---

### `GET /stats`
Get detailed model training statistics.

**Response:**
```json
{
  "model": "BERT (bert-base-cased)",
  "training": {
    "max_epochs": 4,
    "epochs_completed": 3,
    "early_stopping": "enabled",
    "patience": 1,
    "training_time": "1 hour 53 minutes"
  },
  "performance": {
    "accuracy": "82.03%",
    "precision": "82.01%",
    "recall": "82.08%",
    "f1_score": "81.97%"
  }
}
```

---

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

### `GET /docs`
Interactive Swagger UI documentation for testing all endpoints.

---

## Installation

### Prerequisites
- Most preferably Python 3.11.1
- pip package manager

### Local Setup

1. **Clone the repository**
```bash
udnnddgit clone https://github.com/Adedayo2000/news-category-classification.git
cd news-classification
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the API**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

4. **Access the API**
- API: https://adedayo2000-news-category-classification.hf.space/docs#/default/classify_news_classify_post
---

## Docker Deployment

### Build and run with Docker

```bash
# Build the image
docker build -t news-classification-api .

# Run the container
docker run -p 8000:8000 news-classification-api
```

---

## Usage Examples

### Python

```python
import requests

url = "https://adedayo2000-news-category-classification.hf.space/docs#/default/classify_news_classify_post"
data = {
    "text": "The Federal Reserve announced interest rate changes today."
}

response = requests.post(url, json=data)
result = response.json()

print(f"Category: {result['category']}")
print(f"Confidence: {result['confidence']}")
```

### JavaScript

```javascript
fetch('https://adedayo2000-news-category-classification.hf.space/docs#/default/classify_news_classify_post', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    text: 'Scientists discovered a new exoplanet in habitable zone.'
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

### cURL

```bash
curl -X POST "https://adedayo2000-news-category-classification.hf.space/docs#/default/classify_news_classify_post" \
  -H "Content-Type: application/json" \
  -d '{"text": "Lakers won the championship game in overtime."}'
```

---

## 🧠 Model Details

**Base Model:** BERT (bert-base-cased)

**Architecture:**
- Pre-trained BERT encoder (12 layers, 768 hidden units)
- Classification head with 10 output classes
- Total parameters: ~110M

**Training Dataset:** 200K News Classification Dataset
- Source: Kaggle
- Size: 210,800 news articles
- Format: Title + Content combined
- Split: 80% train / 20% test (stratified)

**Model Repository:** 
https://huggingface.co/spaces/Adedayo2000/news-category-classification/tree/main

---

## Training Process

### Configuration

```python
{
  "model_name": "bert-base-cased",
  "test_size": 0.2,
  "random_state": 40,
  "batch_size": 32,
  "epochs": 4,
  "learning_rate": 3e-5,
  "max_length": 128
}
```

### Training Features

**Optimizer:** AdamW
- Learning rate: 3e-5
- Weight decay: 0.01
- Warmup ratio: 0.1

**Early Stopping:**
- Patience: 1 epoch
- Metric: Evaluation loss
- Load best model at end: True

**Why 3 Epochs Instead of 4?**

The model was configured for 4 epochs but stopped at epoch 3 due to **early stopping**. This is a deliberate feature to prevent overfitting:

- **Epoch 1:** Model learning, loss decreasing 
- **Epoch 2:** Continued improvement 
- **Epoch 3:** Validation loss plateaued 
- **Early stopping triggered** → Training stopped

**Benefits:**
-  Prevents overfitting
-  Saves ~30 minutes of training time
-  Optimal model performance
-  Professional ML engineering practice

---

## Performance Metrics

### Test Set Results (Epoch 3)

**Metric & Score**
**Accuracy**: 82.03% 
**Precision**: 82.01% 
**Recall**: 82.08% 
**F1 Score**: 81.97% 

### Training Details

- **Total training time:** 1 hour 53 minutes
- **Epochs completed:** 3 (of 4 max)
- **Platform:** Kaggle GPU (Tesla P100)
- **Dataset size:** 210,800 articles

---

## Deployment

This API is deployed on **Hugging Face Spaces** using Docker.

**Live URL:**https://adedayo2000-news-category-classification.hf.space/docs#/default/classify_news_classify_post

### Deployment Process

1. Model trained on Kaggle GPU (1h 53min)
2. Pushed to Hugging Face Hub (400MB)
3. FastAPI wrapper created
4. Dockerized for deployment
5. Deployed to HF Spaces with automatic rebuilds

### Deployment Features

- **Auto-scaling:** Handles variable traffic
- **Auto-restart:** Recovers from crashes
- **HTTPS:** Secure connections
- **CORS:** Cross-origin requests enabled
- **Documentation:** Auto-generated Swagger UI

---

## Project Structure

```
news-classification/
├── app.py                 # FastAPI application
├── train.py               # Model training script
├── Dockerfile             # Docker configuration
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .gitignore            # Git ignore rules

Model (on Hugging Face Hub):
└── news-classification/
    ├── config.json        # Model configuration
    ├── pytorch_model.bin  # Model weights (400MB)
    └── tokenizer files    # BERT tokenizer
```

---

## What I Learned

Building this project taught me:

### Technical Skills
- **BERT fine-tuning** - Transfer learning for NLP tasks
- **Early stopping** - Preventing overfitting in practice
- **FastAPI** - Building production REST APIs
- **Docker** - Containerizing ML applications
- **HuggingFace Hub** - Model versioning and deployment

---

## Future Improvements

- [ ] **Batch prediction endpoint** - Classify multiple articles at once
- [ ] **Multi-language support** - Extend to non-English news
- [ ] **Fine-grained categories** - Sub-categories (e.g., Tech → AI, Gadgets)
- [ ] **Explainability** - Highlight words that influenced classification
- [ ] **Confidence calibration** - More reliable probability estimates
- [ ] **A/B testing** - Compare different model versions
- [ ] **Caching** - Speed up repeated queries
- [ ] **Rate limiting** - Prevent API abuse
- [ ] **Monitoring dashboard** - Track API usage and performance
- [ ] **Feedback loop** - Allow users to report misclassifications

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

--
---

## 👤 Author

**Adedayo Adebayo**

- Hugging Face: [@Adedayo2000](https://huggingface.co/Adedayo2000)
- GitHub: [@Adedayo2000](https://github.com/Adedayo2000)
- LinkedIn: [Connect with me](https://linkedin.com/in/your-profile)

---

## Acknowledgments

- [Hugging Face](https://huggingface.co/) - For Transformers library and hosting
  
- [FastAPI](https://fastapi.tiangolo.com/) - For the excellent web framework
  
- [Kaggle](https://www.kaggle.com/) - For GPU compute and dataset hosting
  
- [200K News Dataset](https://www.kaggle.com/datasets/adedayoadebayo23/200k-news-classification-dataset) - Training data source

---

##  Contact

Have questions or suggestions? Feel free to reach out!

- **GitHub Issues:** [Create an issue](https://github.com/Adedayo2000/news-category-classification/issues)
- **Email:** adebayoadedayo23@gmail.com

---

## Show Your Support

If you found this project helpful, please give it a ⭐ on GitHub!

---

**Built with ❤️ by Adedayo Adebayo**

*Powered by BERT, FastAPI, and Docker*

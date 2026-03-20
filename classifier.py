import torch
from transformers import BertTokenizer, BertForSequenceClassification
MODEL_PATH = "Adedayo2000/news-category-classification"
def load_tokenizer_and_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    device = torch.device("cpu")
    model.to(device)
    return tokenizer, model, device

def predict_category(news:str,tokenizer,model, device):
    inputs = tokenizer(
        news ,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    new_inputs = {"the first and only time putin and zelensky have ever met in person"}
    for key, value in inputs.items():
        new_inputs[key] = value.to(device)
    inputs = new_inputs
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    predicted_class_id = torch.argmax(probs, dim=1).item()
    confidence = probs[0][predicted_class_id].item()
    category_map={
            "Technology": 0,
            "Politics": 1,
            "Business":2,
            "Education":3,
            "Lifestyle":4,
            "Science":5,
            "Sports":6,
            "Entertainment":7,
            "Health":8,
            "War":9
        }
    prediction = category_map[predicted_class_id]
    print(f"News classification result: {prediction}, with confidence level of: {confidence:.2f}")
    return prediction, confidence
if __name__=="__main__":
    tokenizer, model, device=load_tokenizer_and_model()
    news=""
    predict_category(news, tokenizer,model,device)

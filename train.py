import pandas as pd
import logging 
import torch
import os
import numpy as np 
import shutil
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from huggingface_hub import HfApi, login
from transformers import BertTokenizer,BertForSequenceClassification,Trainer,TrainingArguments,EarlyStoppingCallback
from sklearn.model_selection import train_test_split 
from datasets import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO,format='%(asctime)s--%(levelname)s--%(message)s')
logger=logging.getLogger(__name__)

config={
    "model_name":"bert-base-cased",
    "test_size":0.2,
    "random_state":40,
    "batch_size":32,
    "epochs":4,
    "learning_rate":3e-5,
    "max_length":128
}

def load_and_preprocess():
    try:
        logger.info("Loading news classification dataset...")
        df = pd.read_csv("/kaggle/input/datasets/adedayoadebayo23/200k-news-classification-dataset/200k_news_category.csv")
        if df.columns[0].lower().startswith('unnamed') or 'index' in df.columns[0].lower():
            df = df.drop(df.columns[0], axis=1)
        required_cols = ['title', 'content', 'category']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")
        df["news"] = df["title"].fillna("") + " " + df["content"].fillna("")
        category_map = {
            "Technology": 0,
            "Politics": 1,
            "Business": 2,
            "Education": 3,
            "Lifestyle": 4,
            "Science": 5,
            "Sports": 6,
            "Entertainment": 7,
            "Health": 8,
            "War": 9
        }
        invalid_cats = df[~df["category"].isin(category_map.keys())]["category"].unique()
        if len(invalid_cats) > 0:
            raise ValueError(f"Unrecognized categories found: {invalid_cats.tolist()}")
        df["category"] = df["category"].map(category_map)
        if df["category"].isnull().any():
            raise ValueError("Unmapped category(s) found after mapping")
        logger.info(f"Dataset has {df.shape[0]} rows")
        news = df["news"].tolist()
        category = df["category"].tolist()
        return news, category 
    except Exception as e:
        logger.error(f"Failed to load news classification data due to the following error: {e}")
        raise

def train_test(news:list[str],category: list[int])->tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]:
    try:
        logger.info("Preparing the train_test_split...")
        news_train,news_test,category_train,category_test = train_test_split(
            news, category,
            test_size=config["test_size"],
            random_state=config["random_state"],
            stratify=category
        )
        return news_train,news_test,category_train, category_test
    except Exception as e:
        logger.error(f"An error occurred during train test split:{e}")
        raise

def bert_tokenization(news_train: pd.DataFrame, news_test: pd.DataFrame):
    try:
        logger.info("Tokenizing text...")
        tokenizer = BertTokenizer.from_pretrained(config["model_name"])
        train_encodings = tokenizer(news_train, truncation=True, padding=True, max_length=config["max_length"])
        test_encodings = tokenizer(news_test, truncation=True, padding=True, max_length=config["max_length"])
        return train_encodings, test_encodings, tokenizer 
    except Exception as e:
        logger.error(f"An error occurred during BERT tokenization:{e}")
        raise

def build_dataset(encodings, labels): 
    try:
        logger.info("Building dataset...")
        dataset_dict = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "token_type_ids": encodings.get("token_type_ids", None),
            "labels": labels  
        }
        dataset = Dataset.from_dict(dataset_dict)
        return dataset
    except Exception as e:
        logger.error(f"An error occurred while building dataset: {e}")
        raise
        
def compute_metrics(eval_pred):
    logits, category = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(category, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(category, predictions, average='weighted')
    print(confusion_matrix(category, predictions))
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def push_to_huggingface(model, tokenizer, repo_name: str = "news-category-classification") -> str:
    logger.info("Pushing to HuggingFace Hub...")
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        hf_token = user_secrets.get_secret("HF_TOKEN")
        
        login(token=hf_token)
        logger.info("Logged in to Hugging Face")
        
        api = HfApi()
        user_info = api.whoami(token=hf_token)
        username = user_info['name']
        full_repo_name = f"{username}/{repo_name}"
        
        logger.info(f"Uploading to: {full_repo_name}")
        
        model.push_to_hub(repo_name, token=hf_token)
        tokenizer.push_to_hub(repo_name, token=hf_token)
        
        model_url = f"https://huggingface.co/{full_repo_name}"
        
        logger.info("Model Uploaded Successfully!")
        logger.info(f"URL: {model_url}")
        #logger.info(f"To use: AutoModelForSequenceClassification.from_pretrained('{full_repo_name}')")
        return model_url
    except Exception as e:
        logger.error(f"Failed to push: {e}")
        raise
        
def train_model():
    news, category = load_and_preprocess()
    news_train, news_test, category_train, category_test = train_test(news, category)
    train_encodings, test_encodings, tokenizer= bert_tokenization(news_train, news_test)
    train_dataset = build_dataset(train_encodings, category_train)
    test_dataset = build_dataset(test_encodings, category_test)

    model = BertForSequenceClassification.from_pretrained(config["model_name"], num_labels=10).to(device)

    training_args = TrainingArguments(
        output_dir="./final_model",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=config["epochs"],
        logging_steps=100,
        warmup_ratio=0.1,
        weight_decay=0.01,
        load_best_model_at_end=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Saving model and tokenizer...")
    trainer.save_model("/kaggle/working/final_model")
    tokenizer.save_pretrained("/kaggle/working/final_model")
    logger.info("Training completed.")
    push_to_huggingface(model, tokenizer,"news-category-classification")

if __name__ == "__main__":
    train_model()
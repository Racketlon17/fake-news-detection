import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
import joblib
import os

def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

def load_stopwords():
    download_nltk_data()
    return set(stopwords.words("english"))

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    words = text.split()
    
    stop_words = load_stopwords()
    
    words = [word for word in words if word not in stop_words]
    
    return " ".join(words)

def load_and_preprocess_data(fake_csv_path, true_csv_path):
    # Load datasets
    df_fake = pd.read_csv(fake_csv_path)
    df_true = pd.read_csv(true_csv_path)
    
    df_true["label"] = 1  # True news
    df_fake["label"] = 0  # Fake news
    
    # Combine datasets
    df = pd.concat([df_true, df_fake], ignore_index=True)
    
    df = df[['text', 'label']]

    df["text"] = df["text"].apply(clean_text)
    
    return df

def save_model_components(model, vectorizer, model_dir="model", vectorizer_dir="vectorizer"):
    # Create directories if they don't exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(vectorizer_dir, exist_ok=True)
    
    # Save model and vectorizer
    joblib.dump(model, f"{model_dir}/fake_news_detection_model.pkl")
    joblib.dump(vectorizer, f"{vectorizer_dir}/tfidf_vectorizer.pkl")
    
    print(f"Model saved to {model_dir}/fake_news_detection_model.pkl")
    print(f"Vectorizer saved to {vectorizer_dir}/tfidf_vectorizer.pkl")

def load_model_components(model_path="model/fake_news_detection_model.pkl", 
                         vectorizer_path="vectorizer/tfidf_vectorizer.pkl"):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    return model, vectorizer

def predict_news(text, model, vectorizer):
    cleaned_text = clean_text(text)
    
    text_vector = vectorizer.transform([cleaned_text])
    
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]
    
    label = "Real News" if prediction == 1 else "Fake News"
    confidence = max(probability)
    
    return label, confidence
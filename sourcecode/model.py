import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import load_and_preprocess_data, save_model_components

class FakeNewsDetector:
    def __init__(self, max_df=0.7, test_size=0.2, random_state=42):
        self.max_df = max_df
        self.test_size = test_size
        self.random_state = random_state
        self.vectorizer = None
        self.model = None
        self.is_trained = False
        
    def prepare_data(self, fake_csv_path, true_csv_path):
        print("Loading and preprocessing data...")
        self.df = load_and_preprocess_data(fake_csv_path, true_csv_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Label distribution:\n{self.df['label'].value_counts()}")
        
    def vectorize_text(self):
        print("Vectorizing text data...")
        self.vectorizer = TfidfVectorizer(max_df=self.max_df)
        self.X = self.vectorizer.fit_transform(self.df["text"])
        self.y = self.df["label"]
        print(f"Feature matrix shape: {self.X.shape}")
        
    def split_data(self):
        print("Splitting data...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Testing set size: {self.X_test.shape[0]}")
        
    def train_model(self):
        print("Training model...")
        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.y_train)
        self.is_trained = True
        print("Model training completed!")
        
    def evaluate_model(self):
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
            
        print("Evaluating model...")
        
        self.y_pred = self.model.predict(self.X_test)
        
        accuracy = accuracy_score(self.y_test, self.y_pred)
        report = classification_report(self.y_test, self.y_pred)
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        self.plot_confusion_matrix(cm)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm
        }
        
    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Fake', 'Real'], 
                   yticklabels=['Fake', 'Real'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
        
    def save_model(self, model_dir="model", vectorizer_dir="vectorizer"):
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
            
        save_model_components(self.model, self.vectorizer, model_dir, vectorizer_dir)
        
    def train_complete_pipeline(self, fake_csv_path, true_csv_path):
        self.prepare_data(fake_csv_path, true_csv_path)
        self.vectorize_text()
        self.split_data()
        self.train_model()
        metrics = self.evaluate_model()
        self.save_model()
        
        return metrics

def train_fake_news_model(fake_csv_path="data/Fake.csv", true_csv_path="data/True.csv"):
    detector = FakeNewsDetector()
    detector.train_complete_pipeline(fake_csv_path, true_csv_path)
    return detector
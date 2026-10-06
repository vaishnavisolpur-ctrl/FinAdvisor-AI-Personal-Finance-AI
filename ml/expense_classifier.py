import os
import sys
import re
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)


def clean_text(text: str) -> str:
    """Preprocess transaction description text for NLP."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove reference numbers, transaction IDs, hashes
    text = re.sub(r'#\d+', '', text)
    text = re.sub(r'\bref\b|\btxn\b|\bpos\b', '', text)
    # Remove non-alphanumeric characters except spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    # Collapse multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class ExpenseClassifier:
    """
    ML/NLP Model to classify expense transaction descriptions into 8 categories:
    Food, Transport, Shopping, Bills, Education, Entertainment, Healthcare, Others.
    """
    def __init__(self, model_path=Config.CLASSIFIER_PATH):
        self.model_path = model_path
        self.pipeline = None
        self.classes_ = Config.CATEGORIES
        self.load_model()

    def load_model(self):
        """Loads trained pipeline from disk if available."""
        if os.path.exists(self.model_path):
            try:
                self.pipeline = joblib.load(self.model_path)
                if hasattr(self.pipeline, 'classes_'):
                    self.classes_ = list(self.pipeline.classes_)
                elif hasattr(self.pipeline.named_steps.get('clf'), 'classes_'):
                    self.classes_ = list(self.pipeline.named_steps['clf'].classes_)
            except Exception as e:
                print(f"[ExpenseClassifier] Could not load model: {e}")
                self.pipeline = None

    def train(self, dataset_path=None):
        """
        Trains the TF-IDF + Logistic Regression classification pipeline.
        Computes evaluation metrics (Accuracy, Precision, Recall, F1, Confusion Matrix).
        """
        if dataset_path is None:
            dataset_path = os.path.join(Config.DATASETS_DIR, 'expense_categories_dataset.csv')
            
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Training dataset not found at {dataset_path}")
            
        df = pd.read_csv(dataset_path)
        df['clean_desc'] = df['description'].apply(clean_text)
        df = df[df['clean_desc'].str.len() > 0]
        
        X = df['clean_desc']
        y = df['category']
        
        # 80/20 Stratified Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        
        # Construct NLP Machine Learning Pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=4000,
                sublinear_tf=True
            )),
            ('clf', LogisticRegression(
                C=2.5,
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            ))
        ])
        
        self.pipeline.fit(X_train, y_train)
        self.classes_ = list(self.pipeline.classes_)
        
        # Evaluation
        y_pred = self.pipeline.predict(X_test)
        y_proba = self.pipeline.predict_proba(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=self.classes_)
        
        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)
        
        metrics = {
            'model_name': 'NLP TF-IDF + Logistic Regression Classifier',
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'accuracy': round(float(acc), 4),
            'precision': round(float(prec), 4),
            'recall': round(float(rec), 4),
            'f1_score': round(float(f1), 4),
            'classes': self.classes_,
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        }
        
        print(f"[ExpenseClassifier] Training complete! Test Accuracy: {acc*100:.2f}%, F1: {f1*100:.2f}%")
        return metrics

    def predict(self, description: str):
        """
        Predicts category for a given transaction description.
        Returns: (predicted_category, confidence_score)
        """
        if not self.pipeline:
            return "Others", 0.5
            
        clean_d = clean_text(description)
        if not clean_d:
            return "Others", 0.5
            
        probs = self.pipeline.predict_proba([clean_d])[0]
        max_idx = np.argmax(probs)
        category = self.pipeline.classes_[max_idx]
        confidence = float(probs[max_idx])
        
        return category, round(confidence, 4)

    def predict_all_probabilities(self, description: str):
        """Returns dict of {category: probability}."""
        if not self.pipeline:
            return {c: 1.0 / len(self.classes_) for c in self.classes_}
            
        clean_d = clean_text(description)
        if not clean_d:
            return {c: 1.0 / len(self.classes_) for c in self.classes_}
            
        probs = self.pipeline.predict_proba([clean_d])[0]
        return {cat: round(float(p), 4) for cat, p in zip(self.pipeline.classes_, probs)}

if __name__ == '__main__':
    clf = ExpenseClassifier()
    metrics = clf.train()
    print("Test Predictions:")
    tests = [
        "Swiggy order biryani dinner",
        "Uber ride to international airport",
        "BESCOM electricity monthly bill",
        "Apollo pharmacy paracetamol tablets",
        "Netflix 4K UHD streaming renewal",
        "Bought Nike running shoes from showroom",
        "Coursera Machine Learning certification fee"
    ]
    for t in tests:
        cat, conf = clf.predict(t)
        print(f"'{t}' -> {cat} ({conf*100:.1f}%)")

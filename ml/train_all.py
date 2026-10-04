import os
import sys
import json
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config
from ml.expense_classifier import ExpenseClassifier
from ml.expense_predictor import ExpensePredictor
from ml.anomaly_detector import AnomalyDetector

def train_all_models():
    """
    Master training pipeline that trains all 3 Machine Learning models,
    evaluates them on validation/test sets, and persists them to models/ directory.
    """
    print("=" * 70)
    print(">>> STARTING AI-POWERED PERSONAL FINANCE ADVISOR ML TRAINING PIPELINE")
    print("=" * 70)
    
    start_time = time.time()
    all_metrics = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'environment': 'Scikit-Learn, Pandas, NumPy, Python 3.10',
        'models': {}
    }
    
    # 1. Train Expense Classifier
    print("\n[1/3] Training NLP Expense Categorization Model...")
    clf = ExpenseClassifier()
    clf_metrics = clf.train()
    all_metrics['models']['expense_classifier'] = clf_metrics
    
    # 2. Train Expense Predictor
    print("\n[2/3] Training Monthly Expense Regression Forecaster...")
    predictor = ExpensePredictor()
    pred_metrics = predictor.train()
    all_metrics['models']['expense_predictor'] = pred_metrics
    
    # 3. Train Anomaly Detector
    print("\n[3/3] Training Unsupervised Isolation Forest Anomaly Detector...")
    detector = AnomalyDetector()
    anom_metrics = detector.train(synthetic_samples=5000)
    all_metrics['models']['anomaly_detector'] = anom_metrics
    
    # Save metrics JSON for the Viva/ML Dashboard
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    with open(Config.METRICS_PATH, 'w') as f:
        json.dump(all_metrics, f, indent=4)
        
    duration = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"[SUCCESS] ALL ML MODELS TRAINED & PERSISTED SUCCESSFULLY in {duration:.2f} seconds!")
    print(f"Models Directory: {Config.MODELS_DIR}")
    print(f"Metrics File:     {Config.METRICS_PATH}")
    print("=" * 70)
    
    return all_metrics


if __name__ == '__main__':
    train_all_models()

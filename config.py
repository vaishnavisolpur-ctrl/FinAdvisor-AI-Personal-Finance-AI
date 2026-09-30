import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'fintech-ai-super-secret-key-2026-aiml')
    DATABASE_PATH = os.path.join(BASE_DIR, 'database', 'finance_advisor.db')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')
    REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
    
    # Model file paths
    CLASSIFIER_PATH = os.path.join(MODELS_DIR, 'category_classifier.joblib')
    PREDICTOR_PATH = os.path.join(MODELS_DIR, 'expense_predictor.joblib')
    ANOMALY_PATH = os.path.join(MODELS_DIR, 'anomaly_detector.joblib')
    METRICS_PATH = os.path.join(MODELS_DIR, 'model_metrics.json')
    
    # Financial Categories
    CATEGORIES = [
        'Food',
        'Transport',
        'Shopping',
        'Bills',
        'Education',
        'Entertainment',
        'Healthcare',
        'Others'
    ]
    
    # Payment Methods
    PAYMENT_METHODS = [
        'Credit Card',
        'Debit Card',
        'UPI / Digital Wallet',
        'Net Banking',
        'Cash'
    ]

import os
import sys
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from config import Config

class AnomalyDetector:
    """
    Unsupervised ML Anomaly Detection using Isolation Forest combined with
    category-specific statistical z-scores to detect abnormal or suspicious transactions.
    """
    
    # Baseline benchmark stats per category (amounts in INR)
    # Used when a user is new or has very few transactions in that category
    CATEGORY_BASELINES = {
        'Food': {'mean': 650.0, 'std': 450.0, 'max_normal': 2500.0},
        'Transport': {'mean': 450.0, 'std': 350.0, 'max_normal': 2200.0},
        'Shopping': {'mean': 2200.0, 'std': 1800.0, 'max_normal': 9000.0},
        'Bills': {'mean': 2800.0, 'std': 1900.0, 'max_normal': 10000.0},
        'Education': {'mean': 3500.0, 'std': 3000.0, 'max_normal': 15000.0},
        'Entertainment': {'mean': 800.0, 'std': 600.0, 'max_normal': 3000.0},
        'Healthcare': {'mean': 1500.0, 'std': 1400.0, 'max_normal': 7000.0},
        'Others': {'mean': 1000.0, 'std': 900.0, 'max_normal': 4500.0}
    }

    def __init__(self, model_path=Config.ANOMALY_PATH):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.load_model()

    def load_model(self):
        """Loads trained IsolationForest and scaler."""
        if os.path.exists(self.model_path):
            try:
                bundle = joblib.load(self.model_path)
                self.model = bundle['model']
                self.scaler = bundle['scaler']
            except Exception as e:
                print(f"[AnomalyDetector] Could not load model: {e}")
                self.model = None
                self.scaler = None

    def train(self, synthetic_samples=5000):
        """
        Trains an Isolation Forest on a distribution of normal transactions
        plus a 4% synthetic contamination of high-value outliers.
        """
        np.random.seed(42)
        records = []
        
        # 1. Normal transactions (96%)
        num_normal = int(synthetic_samples * 0.96)
        categories = Config.CATEGORIES
        
        for _ in range(num_normal):
            cat = np.random.choice(categories)
            base = self.CATEGORY_BASELINES[cat]
            # Log-normal distribution to mimic real spend
            amount = np.random.exponential(base['mean']) + 50.0
            amount = min(amount, base['max_normal'] * 1.2)
            cat_idx = categories.index(cat)
            ratio = amount / base['mean']
            records.append([amount, cat_idx, ratio])
            
        # 2. Injected Anomalies (4%)
        num_anom = synthetic_samples - num_normal
        for _ in range(num_anom):
            cat = np.random.choice(categories)
            base = self.CATEGORY_BASELINES[cat]
            # 3x to 8x the max normal amount
            amount = base['max_normal'] * np.random.uniform(2.5, 7.0)
            cat_idx = categories.index(cat)
            ratio = amount / base['mean']
            records.append([amount, cat_idx, ratio])
            
        X = np.array(records)
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Isolation Forest with 4% contamination
        self.model = IsolationForest(
            n_estimators=120,
            contamination=0.04,
            max_samples='auto',
            random_state=42
        )
        self.model.fit(X_scaled)
        
        # Predictions on training data
        preds = self.model.predict(X_scaled)  # -1 = anomaly, 1 = normal
        scores = self.model.decision_function(X_scaled)
        
        anom_detected = int(np.sum(preds == -1))
        
        # Save bundle
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump({'model': self.model, 'scaler': self.scaler}, self.model_path)
        
        metrics = {
            'model_name': 'Isolation Forest Anomaly Detector',
            'total_samples': synthetic_samples,
            'contamination_rate': 0.04,
            'anomalies_flagged': anom_detected,
            'normal_flagged': synthetic_samples - anom_detected,
            'min_decision_score': round(float(np.min(scores)), 4),
            'max_decision_score': round(float(np.max(scores)), 4),
            'mean_decision_score': round(float(np.mean(scores)), 4)
        }
        
        print(f"[AnomalyDetector] Training complete! Contamination rate: 4.0%, Flagged: {anom_detected}/{synthetic_samples}")
        return metrics

    def detect(self, amount: float, category: str, user_category_stats: dict = None):
        """
        Determines whether a transaction is anomalous.
        
        Returns:
            dict with {
                'is_anomaly': bool,
                'severity': 'None' | 'Moderate' | 'High',
                'anomaly_score': float,
                'ratio_to_avg': float,
                'category_avg': float,
                'warning': str
            }
        """
        if category not in Config.CATEGORIES:
            category = 'Others'
            
        amount = float(amount)
        cat_idx = Config.CATEGORIES.index(category)
        
        # Determine category baseline: user's personal history if available, else system baseline
        if user_category_stats and category in user_category_stats and user_category_stats[category]['count'] >= 3:
            stats = user_category_stats[category]
            cat_mean = float(stats['mean'])
            cat_std = max(100.0, float(stats['std']))
        else:
            base = self.CATEGORY_BASELINES.get(category, self.CATEGORY_BASELINES['Others'])
            cat_mean = base['mean']
            cat_std = base['std']
            
        ratio = amount / cat_mean if cat_mean > 0 else 1.0
        z_score = (amount - cat_mean) / cat_std if cat_std > 0 else 0.0
        
        # Isolation Forest scoring
        ml_is_anom = False
        decision_score = 0.5
        if self.model and self.scaler:
            feat = np.array([[amount, cat_idx, ratio]])
            feat_scaled = self.scaler.transform(feat)
            decision_score = float(self.model.decision_function(feat_scaled)[0])
            ml_pred = self.model.predict(feat_scaled)[0]
            ml_is_anom = (ml_pred == -1)
            
        # Statistical rule threshold: ratio >= 2.8 or z-score >= 2.5
        stat_is_anom = (ratio >= 2.6 and amount > cat_mean + 1000.0) or (z_score >= 2.6)
        
        # Combine ML + Statistical
        is_anomaly = ml_is_anom or stat_is_anom
        
        if not is_anomaly:
            return {
                'is_anomaly': False,
                'severity': 'None',
                'anomaly_score': round(decision_score, 3),
                'ratio_to_avg': round(ratio, 2),
                'category_avg': round(cat_mean, 2),
                'warning': None
            }
            
        # Determine severity & explanation
        if ratio >= 4.0 or z_score >= 3.5:
            severity = 'High'
            warning = f"High Anomaly: This transaction of ₹{amount:,.2f} is {ratio:.1f}x higher than your usual {category} spending (avg ₹{cat_mean:,.2f})."
        else:
            severity = 'Moderate'
            warning = f"Notice: This transaction is significantly higher than your usual spending in {category} (avg ₹{cat_mean:,.2f})."
            
        return {
            'is_anomaly': True,
            'severity': severity,
            'anomaly_score': round(decision_score, 3),
            'ratio_to_avg': round(ratio, 2),
            'category_avg': round(cat_mean, 2),
            'warning': warning
        }

if __name__ == '__main__':
    detector = AnomalyDetector()
    metrics = detector.train()
    print("Testing Anomaly Detector:")
    test_cases = [
        (450.0, 'Food'),
        (12000.0, 'Food'),      # Huge food bill!
        (350.0, 'Transport'),
        (18500.0, 'Transport'), # Huge flight or car expense
        (2500.0, 'Shopping'),
        (48000.0, 'Shopping')   # Sudden massive shopping
    ]
    for amt, cat in test_cases:
        res = detector.detect(amt, cat)
        print(f"₹{amt:,.2f} in {cat} -> Anomaly: {res['is_anomaly']} ({res['severity']}) : {res['warning']}")

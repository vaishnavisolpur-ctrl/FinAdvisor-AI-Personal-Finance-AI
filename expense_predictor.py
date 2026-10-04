import os
import sys
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import Config

class ExpensePredictor:
    """
    ML Regression Model to forecast future monthly expenses based on:
    - Prior month expenses (lag-1, lag-2, lag-3)
    - 3-month moving average and spending volatility (std)
    - User's monthly income
    - Essential spending ratio (Food + Bills + Healthcare)
    - Seasonal cyclical signals (Sine/Cosine month indicators)
    """
    
    FEATURE_COLS = [
        'monthly_income',
        'lag_1_expense',
        'lag_2_expense',
        'lag_3_expense',
        'rolling_3m_mean',
        'rolling_3m_std',
        'essential_ratio',
        'month_sin',
        'month_cos'
    ]
    
    def __init__(self, model_path=Config.PREDICTOR_PATH):
        self.model_path = model_path
        self.pipeline = None
        self.load_model()

    def load_model(self):
        """Loads trained regressor from disk if available."""
        if os.path.exists(self.model_path):
            try:
                self.pipeline = joblib.load(self.model_path)
            except Exception as e:
                print(f"[ExpensePredictor] Could not load model: {e}")
                self.pipeline = None

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds cyclical trigonometric encoding for months."""
        df = df.copy()
        month = df['month_of_year']
        df['month_sin'] = np.sin(2 * np.pi * month / 12.0)
        df['month_cos'] = np.cos(2 * np.pi * month / 12.0)
        return df

    def train(self, dataset_path=None):
        """
        Trains the Expense Predictor using RandomForestRegressor with StandardScaler.
        Calculates MAE, RMSE, and R2 score.
        """
        if dataset_path is None:
            dataset_path = os.path.join(Config.DATASETS_DIR, 'historical_monthly_expenses.csv')
            
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Regression dataset not found at {dataset_path}")
            
        df = pd.read_csv(dataset_path)
        df = self._engineer_features(df)
        
        X = df[self.FEATURE_COLS]
        y = df['actual_expense']
        
        # 80/20 Train-Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )
        
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                min_samples_split=4,
                random_state=42
            ))
        ])
        
        self.pipeline.fit(X_train, y_train)
        
        # Predictions on holdout test set
        y_pred = self.pipeline.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Feature importances
        rf_model = self.pipeline.named_steps['regressor']
        importances = dict(zip(self.FEATURE_COLS, [round(float(v), 4) for v in rf_model.feature_importances_]))
        
        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)
        
        metrics = {
            'model_name': 'Random Forest Expense Regressor',
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'mae': round(float(mae), 2),
            'rmse': round(float(rmse), 2),
            'r2_score': round(float(r2), 4),
            'mape_pct': round(float(mape), 2),
            'feature_importances': importances,
            'test_actuals_sample': [round(float(v), 2) for v in y_test.iloc[:10].tolist()],
            'test_preds_sample': [round(float(v), 2) for v in y_pred[:10].tolist()]
        }
        
        print(f"[ExpensePredictor] Training complete! MAE: Rs. {mae:.2f}, RMSE: Rs. {rmse:.2f}, R2: {r2:.4f}, MAPE: {mape:.2f}%")
        return metrics

    def predict_next_month(self, monthly_summary_list: list, current_income: float, next_month: int = None):
        """
        Takes recent monthly totals and current income to predict next month's expenses.
        
        monthly_summary_list: list of dicts ordered chronologically e.g.:
          [{'month': 7, 'total_expense': 34000, 'essential_expense': 21000}, ...]
        """
        import datetime
        now = datetime.datetime.now()
        if next_month is None:
            next_month = (now.month % 12) + 1
            
        # Default baseline if limited user history exists (< 3 months)
        if not monthly_summary_list or len(monthly_summary_list) == 0:
            # Simple rule-of-thumb baseline: 65% of income
            baseline = current_income * 0.65 if current_income > 0 else 25000.0
            return {
                'predicted_expense': round(baseline, 2),
                'lower_bound': round(baseline * 0.88, 2),
                'upper_bound': round(baseline * 1.12, 2),
                'confidence': 'Low (Requires at least 2-3 months of data)',
                'trend': 'Stable',
                'explanation': 'Baseline heuristic projection based on current income until more transaction history is recorded.'
            }
            
        # Extract lags
        expenses = [m.get('total_expense', m.get('expense', 0.0)) for m in monthly_summary_list]
        essentials = [m.get('essential_expense', m.get('expense', 0.0) * 0.6) for m in monthly_summary_list]

        
        lag_1 = expenses[-1]
        lag_2 = expenses[-2] if len(expenses) >= 2 else lag_1
        lag_3 = expenses[-3] if len(expenses) >= 3 else lag_2
        
        recent_window = expenses[-3:] if len(expenses) >= 3 else expenses
        rolling_3m = float(np.mean(recent_window))
        rolling_std = float(np.std(recent_window)) if len(recent_window) > 1 else 1500.0
        
        last_essential = essentials[-1]
        essential_ratio = (last_essential / lag_1) if lag_1 > 0 else 0.55
        essential_ratio = max(0.2, min(0.9, essential_ratio))
        
        income_val = current_income if current_income > 0 else (rolling_3m * 1.3)
        
        # Prepare feature vector
        row = {
            'monthly_income': income_val,
            'lag_1_expense': lag_1,
            'lag_2_expense': lag_2,
            'lag_3_expense': lag_3,
            'rolling_3m_mean': rolling_3m,
            'rolling_3m_std': rolling_std,
            'essential_ratio': essential_ratio,
            'month_sin': np.sin(2 * np.pi * next_month / 12.0),
            'month_cos': np.cos(2 * np.pi * next_month / 12.0)
        }
        
        if self.pipeline:
            feat_df = pd.DataFrame([row])[self.FEATURE_COLS]
            pred = float(self.pipeline.predict(feat_df)[0])
        else:
            # Fallback weighted moving average
            pred = (0.5 * lag_1) + (0.3 * rolling_3m) + (0.2 * (income_val * 0.65))
            
        # Safety bounds
        pred = max(500.0, pred)
        margin = max(1200.0, rolling_std * 1.2)
        lower_bound = max(0.0, pred - margin)
        upper_bound = pred + margin
        
        # Trend comparison
        diff = pred - lag_1
        pct_diff = (diff / lag_1 * 100) if lag_1 > 0 else 0
        if pct_diff > 4:
            trend = "Increasing"
            explanation = f"Projected to increase by ₹{diff:,.2f} (+{pct_diff:.1f}%) compared to last month due to recent spending velocity and seasonal trends."
        elif pct_diff < -4:
            trend = "Decreasing"
            explanation = f"Projected to decrease by ₹{abs(diff):,.2f} ({pct_diff:.1f}%) compared to last month, reflecting positive consolidation."
        else:
            trend = "Stable"
            explanation = f"Projected to remain relatively consistent with last month's spending within a ±4% variance margin."
            
        confidence_level = "High" if len(expenses) >= 6 else ("Medium" if len(expenses) >= 3 else "Moderate")
        
        return {
            'predicted_expense': round(pred, 2),
            'lower_bound': round(lower_bound, 2),
            'upper_bound': round(upper_bound, 2),
            'confidence': confidence_level,
            'trend': trend,
            'explanation': explanation,
            'lag_1': round(lag_1, 2),
            'rolling_3m': round(rolling_3m, 2),
            'margin': round(margin, 2)
        }

if __name__ == '__main__':
    predictor = ExpensePredictor()
    metrics = predictor.train()
    print("Test Sample Prediction:")
    sample_history = [
        {'month': 1, 'total_expense': 38000, 'essential_expense': 22000},
        {'month': 2, 'total_expense': 41000, 'essential_expense': 24000},
        {'month': 3, 'total_expense': 39500, 'essential_expense': 23000}
    ]
    res = predictor.predict_next_month(sample_history, current_income=65000, next_month=4)
    print(res)

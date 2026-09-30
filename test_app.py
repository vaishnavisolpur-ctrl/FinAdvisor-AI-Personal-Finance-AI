import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from app import app
from database.db import init_db
from database.seed_demo_data import seed_demo_user

class FinanceAdvisorTestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()
        init_db()
        seed_demo_user(force=False)

    def test_01_public_pages(self):
        """Test login and register pages load with 200 OK."""
        res_login = self.client.get('/login')
        self.assertEqual(res_login.status_code, 200)
        self.assertIn(b'Sign In', res_login.data)

        res_reg = self.client.get('/register')
        self.assertEqual(res_reg.status_code, 200)
        self.assertIn(b'Create Account', res_reg.data)

    def test_02_demo_login_and_dashboard(self):
        """Test 1-click demo login and redirect to dashboard."""
        res = self.client.get('/login/demo', follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Financial Intelligence Dashboard', res.data)
        self.assertIn(b'Alex Sharma', res.data)
        self.assertIn(b'Predicted Next Month', res.data)

    def test_03_api_predict_category(self):
        """Test real-time NLP classification API."""
        with self.client:
            self.client.get('/login/demo')
            res = self.client.get('/api/predict-category?desc=Swiggy+order+chicken+biryani')
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertEqual(data['category'], 'Food')
            self.assertGreater(data['confidence'], 80)

            res2 = self.client.get('/api/predict-category?desc=Uber+cab+ride+to+airport')
            data2 = res2.get_json()
            self.assertEqual(data2['category'], 'Transport')

            res3 = self.client.get('/api/predict-category?desc=BESCOM+electricity+bill')
            data3 = res3.get_json()
            self.assertEqual(data3['category'], 'Bills')

    def test_04_api_anomaly_check(self):
        """Test real-time Anomaly Detection API."""
        with self.client:
            self.client.get('/login/demo')
            # Normal amount in Food
            res_normal = self.client.post('/api/check-anomaly', json={'amount': 450, 'category': 'Food'})
            self.assertFalse(res_normal.get_json()['is_anomaly'])

            # Massive amount in Food (₹18,000)
            res_anom = self.client.post('/api/check-anomaly', json={'amount': 18000, 'category': 'Food'})
            self.assertTrue(res_anom.get_json()['is_anomaly'])

    def test_05_authenticated_views(self):
        """Test all core views load with 200 OK for logged-in user."""
        with self.client:
            self.client.get('/login/demo')
            
            pages = [
                ('/transactions', b'Transaction Ledger'),
                ('/add-transaction', b'New Transaction Entry'),
                ('/expense-prediction', b'ML Future Expense Forecast'),
                ('/ai-insights', b'AI Financial Insights'),
                ('/budget-recommendation', b'Personalized Budget Recommendation'),
                ('/savings-goals', b'Savings Goal Milestones'),
                ('/ml-models', b'Machine Learning Architecture'),
                ('/reports', b'Monthly Financial Report'),
                ('/profile', b'User Profile')
            ]
            for url, keyword in pages:
                res = self.client.get(url)
                self.assertEqual(res.status_code, 200, f"Page {url} failed with {res.status_code}")
                self.assertIn(keyword, res.data, f"Page {url} missing keyword")

    def test_06_add_transaction_flow(self):
        """Test adding a transaction with automatic categorization."""
        with self.client:
            self.client.get('/login/demo')
            res = self.client.post('/add-transaction', data={
                'type': 'expense',
                'amount': '350.00',
                'category': 'Auto',
                'payment_method': 'UPI / Digital Wallet',
                'date': '2026-09-30',
                'description': 'Starbucks tall cappuccino with oat milk'
            }, follow_redirects=True)
            self.assertEqual(res.status_code, 200)
            self.assertIn(b'Transaction recorded successfully', res.data)

    def test_07_csv_export(self):
        """Test exporting transactions CSV."""
        with self.client:
            self.client.get('/login/demo')
            res = self.client.get('/export/csv')
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.content_type, 'text/csv; charset=utf-8')
            self.assertIn(b'Amount (INR)', res.data)

if __name__ == '__main__':
    unittest.main()

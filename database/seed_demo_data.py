import os
import sys
import datetime
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config
from database.db import (
    init_db, get_db_connection, create_user, 
    add_transaction, create_savings_goal
)
from ml.anomaly_detector import AnomalyDetector
from ml.expense_classifier import ExpenseClassifier

def seed_demo_user(force=False):
    """
    Seeds a rich demonstration account with 12 months of realistic transactions,
    savings goals, realistic anomalies, and income entries.
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if demo user exists
    cursor.execute("SELECT id FROM users WHERE email = 'demo@finance.ai'")
    existing = cursor.fetchone()
    
    if existing:
        if not force:
            print("[Seed] Demo user 'demo@finance.ai' already exists. Skipping seed.")
            conn.close()
            return existing['id']
        else:
            print("[Seed] Force flag enabled: resetting demo user transactions & goals...")
            u_id = existing['id']
            cursor.execute("DELETE FROM transactions WHERE user_id = ?", (u_id,))
            cursor.execute("DELETE FROM savings_goals WHERE user_id = ?", (u_id,))
            conn.commit()
    else:
        u_id, err = create_user(
            name="Alex Sharma",
            email="demo@finance.ai",
            password="password123",
            monthly_income=75000.0,
            currency="₹"
        )
        print(f"[Seed] Created demo user (ID: {u_id}) with email: demo@finance.ai")
        
    conn.close()
    
    # ML Models for tagging
    classifier = ExpenseClassifier()
    detector = AnomalyDetector()
    
    # Generate 12 months of historical transactions leading up to current date
    now = datetime.datetime.now()
    
    # Curated recurring templates
    monthly_templates = [
        # Essentials
        ("BESCOM Electricity Bill payment", "Bills", 1850.0, "Net Banking"),
        ("Airtel Broadband Fiber Internet 200Mbps", "Bills", 1180.0, "Net Banking"),
        ("House Maintenance and Society Dues", "Bills", 3200.0, "Net Banking"),
        ("Jio 5G Mobile Postpaid Unlimited Plan", "Bills", 599.0, "UPI / Digital Wallet"),
        ("Nature's Basket Weekly Organic Grocery", "Food", 2450.0, "Debit Card"),
        ("D-Mart Supermarket Monthly Provisions", "Food", 4800.0, "Credit Card"),
        ("Swiggy Biryani Feast with friends", "Food", 850.0, "UPI / Digital Wallet"),
        ("Zomato Gourmet Pizza Delivery", "Food", 720.0, "UPI / Digital Wallet"),
        ("Starbucks Hazelnut Latte & Croissant", "Food", 480.0, "Credit Card"),
        ("Daily milk, bread & eggs via Blinkit", "Food", 320.0, "UPI / Digital Wallet"),
        ("Shell Petrol station fuel car tank", "Transport", 2800.0, "Credit Card"),
        ("Uber cab rides to tech park office", "Transport", 1450.0, "UPI / Digital Wallet"),
        ("Namma Metro Card Monthly Recharge", "Transport", 600.0, "UPI / Digital Wallet"),
        ("FASTag Highway Toll Automatic Recharge", "Transport", 500.0, "Net Banking"),
        ("Apollo Pharmacy Vitamin D & multivitamins", "Healthcare", 890.0, "Debit Card"),
        ("Dental checkup and tartar cleaning", "Healthcare", 1200.0, "UPI / Digital Wallet"),
        # Discretionary
        ("Netflix 4K UHD Monthly Subscription", "Entertainment", 649.0, "Credit Card"),
        ("Spotify Premium Individual Music Plan", "Entertainment", 119.0, "UPI / Digital Wallet"),
        ("BookMyShow IMAX 3D movie tickets with snacks", "Entertainment", 950.0, "Credit Card"),
        ("Amazon India gadgets and desk organizer", "Shopping", 1650.0, "Credit Card"),
        ("Myntra casual polo shirts and socks", "Shopping", 2100.0, "Credit Card"),
        ("Zara cotton slim fit trousers", "Shopping", 3290.0, "Credit Card"),
        ("Udemy Python Machine Learning course", "Education", 499.0, "UPI / Digital Wallet"),
        ("Medium / Substack tech publication annual", "Education", 850.0, "Credit Card"),
        ("Laundry and dry cleaning suit blazer", "Others", 750.0, "Cash"),
        ("Car wash spa and interior detailing", "Others", 650.0, "UPI / Digital Wallet")
    ]
    
    # 4 Notable Anomalies to inject realistically across specific months
    anomaly_specials = [
        {
            'month_offset': 8,
            'desc': 'Emergency Apollo Hospital Surgery & Diagnostics',
            'category': 'Healthcare',
            'amount': 32500.0,
            'method': 'Credit Card',
            'reason': 'High Anomaly: This transaction of ₹32,500.00 is 21.7x higher than your usual Healthcare spending (avg ₹1,500.00).'
        },
        {
            'month_offset': 5,
            'desc': 'Grand Five-Star Luxury Anniversary Dinner at The Leela',
            'category': 'Food',
            'amount': 14800.0,
            'method': 'Credit Card',
            'reason': 'High Anomaly: This transaction of ₹14,800.00 is 17.4x higher than your usual Food spending (avg ₹850.00).'
        },
        {
            'month_offset': 3,
            'desc': 'Apple Store 4K UltraSharp External Monitor and Ergonomic Chair',
            'category': 'Shopping',
            'amount': 43500.0,
            'method': 'Credit Card',
            'reason': 'High Anomaly: This transaction of ₹43,500.00 is 19.8x higher than your usual Shopping spending (avg ₹2,200.00).'
        },
        {
            'month_offset': 1,
            'desc': 'Urgent Round-trip Flight Tickets to Delhi for family event',
            'category': 'Transport',
            'amount': 18400.0,
            'method': 'Net Banking',
            'reason': 'High Anomaly: This transaction of ₹18,400.00 is 12.7x higher than your usual Transport spending (avg ₹1,450.00).'
        }
    ]
    
    # Seed 12 months (month 11 ago down to month 0 = current month)
    total_added = 0
    for offset in range(11, -1, -1):
        # Calculate target month date
        # If offset is 0, it's current month; if offset is 1, it's last month, etc.
        m_year = now.year
        m_month = now.month - offset
        while m_month <= 0:
            m_month += 12
            m_year -= 1
            
        # 1. Add Monthly Salary on 1st of month
        sal_date = f"{m_year:04d}-{m_month:02d}-01"
        add_transaction(
            user_id=u_id,
            trans_type='income',
            amount=75000.0,
            category='Others',
            payment_method='Net Banking',
            date=sal_date,
            description="Monthly Corporate Salary Credit - Infosys Ltd"
        )
        total_added += 1
        
        # Quarterly bonus / freelance income in months 2, 5, 8, 11
        if offset in [2, 5, 8, 11]:
            bonus_date = f"{m_year:04d}-{m_month:02d}-15"
            bonus_amt = 25000.0 if offset % 2 == 0 else 32000.0
            add_transaction(
                user_id=u_id,
                trans_type='income',
                amount=bonus_amt,
                category='Others',
                payment_method='Net Banking',
                date=bonus_date,
                description="Freelance AI Consulting / Quarterly Performance Incentive"
            )
            total_added += 1
            
        # Pick 14-18 expenses for this month
        sample_size = random.randint(14, 18)
        selected_expenses = random.sample(monthly_templates, sample_size)
        
        for desc, cat, base_amt, pmethod in selected_expenses:
            # Vary amount slightly (+/- 10%)
            amt = round(base_amt * random.uniform(0.92, 1.08), 2)
            day = min(28, random.randint(2, 28))
            txn_date = f"{m_year:04d}-{m_month:02d}-{day:02d}"
            
            # Predict or verify category via classifier
            predicted_cat, conf = classifier.predict(desc)
            final_cat = cat if cat else predicted_cat
            
            # Anomaly check
            det_res = detector.detect(amt, final_cat)
            
            add_transaction(
                user_id=u_id,
                trans_type='expense',
                amount=amt,
                category=final_cat,
                payment_method=pmethod,
                date=txn_date,
                description=desc,
                is_anomaly=1 if det_res['is_anomaly'] else 0,
                anomaly_score=det_res['anomaly_score'],
                anomaly_reason=det_res['warning']
            )
            total_added += 1
            
        # Inject designated anomaly if month matches
        for anom in anomaly_specials:
            if anom['month_offset'] == offset:
                day = min(25, random.randint(8, 24))
                txn_date = f"{m_year:04d}-{m_month:02d}-{day:02d}"
                add_transaction(
                    user_id=u_id,
                    trans_type='expense',
                    amount=anom['amount'],
                    category=anom['category'],
                    payment_method=anom['method'],
                    date=txn_date,
                    description=anom['desc'],
                    is_anomaly=1,
                    anomaly_score=-0.38,
                    anomaly_reason=anom['reason']
                )
                total_added += 1
                
    # 3. Create Savings Goals
    create_savings_goal(
        user_id=u_id,
        title="Emergency Contingency Fund",
        target_amount=150000.0,
        current_amount=112000.0,
        target_date="2026-12-31",
        category="Emergency Fund"
    )
    create_savings_goal(
        user_id=u_id,
        title="Apple MacBook Pro M3 Setup",
        target_amount=125000.0,
        current_amount=82000.0,
        target_date="2026-11-20",
        category="Electronics / Education"
    )
    create_savings_goal(
        user_id=u_id,
        title="Japan Cherry Blossom Spring Vacation",
        target_amount=200000.0,
        current_amount=65000.0,
        target_date="2027-04-15",
        category="Travel & Leisure"
    )
    
    print(f"[Seed] Successfully populated {total_added} realistic historical transactions and 3 savings goals for demo user!")
    return u_id

if __name__ == '__main__':
    seed_demo_user(force=True)

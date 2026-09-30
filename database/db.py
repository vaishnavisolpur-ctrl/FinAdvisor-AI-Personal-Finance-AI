import os
import sqlite3
import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from config import Config

def get_db_connection():
    """Returns a SQLite connection with dictionary-like row factory."""
    os.makedirs(os.path.dirname(Config.DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(Config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    """Initializes SQLite database schema and required indexes."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        monthly_income REAL DEFAULT 50000.0,
        currency TEXT DEFAULT '₹',
        occupation TEXT DEFAULT 'Professional / Student',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # 2. Transactions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        type TEXT CHECK(type IN ('income', 'expense')) NOT NULL,
        amount REAL NOT NULL,
        category TEXT NOT NULL,
        payment_method TEXT NOT NULL,
        date TEXT NOT NULL,
        description TEXT NOT NULL,
        is_anomaly INTEGER DEFAULT 0,
        anomaly_score REAL DEFAULT 0.0,
        anomaly_reason TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """)
    
    # 3. Savings Goals table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS savings_goals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        target_amount REAL NOT NULL,
        current_amount REAL DEFAULT 0.0,
        target_date TEXT,
        category TEXT DEFAULT 'General Savings',
        status TEXT CHECK(status IN ('active', 'completed')) DEFAULT 'active',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """)
    
    # 4. Monthly Budgets table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS category_budgets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        category TEXT NOT NULL,
        monthly_limit REAL NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(user_id, category),
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """)
    
    # Indexes for fast querying
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trans_user_date ON transactions(user_id, date);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trans_user_cat ON transactions(user_id, category);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_goals_user ON savings_goals(user_id, status);")
    
    conn.commit()
    conn.close()
    print(f"[Database] SQLite schema verified and ready at: {Config.DATABASE_PATH}")

# ================= USER OPERATIONS =================

def create_user(name, email, password, monthly_income=50000.0, currency='₹'):
    """Creates a new user with hashed password."""
    conn = get_db_connection()
    cursor = conn.cursor()
    p_hash = generate_password_hash(password)
    try:
        cursor.execute(
            "INSERT INTO users (name, email, password_hash, monthly_income, currency) VALUES (?, ?, ?, ?, ?)",
            (name.strip(), email.strip().lower(), p_hash, float(monthly_income), currency)
        )
        conn.commit()
        user_id = cursor.lastrowid
        return user_id, None
    except sqlite3.IntegrityError:
        return None, "An account with this email address already exists."
    finally:
        conn.close()

def authenticate_user(email, password):
    """Authenticates user credentials; returns user dict or None."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),))
    user = cursor.fetchone()
    conn.close()
    if user and check_password_hash(user['password_hash'], password):
        return dict(user)
    return None

def get_user_by_id(user_id):
    """Fetches user details by user_id."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email, monthly_income, currency, occupation, created_at FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()
    return dict(user) if user else None

def update_user_profile(user_id, name, monthly_income, occupation):
    """Updates user profile settings."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE users SET name = ?, monthly_income = ?, occupation = ? WHERE id = ?",
        (name.strip(), float(monthly_income), occupation.strip(), user_id)
    )
    conn.commit()
    conn.close()

# ================= TRANSACTION OPERATIONS =================

def add_transaction(user_id, trans_type, amount, category, payment_method, date, description, is_anomaly=0, anomaly_score=0.0, anomaly_reason=None):
    """Inserts a new transaction."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO transactions 
        (user_id, type, amount, category, payment_method, date, description, is_anomaly, anomaly_score, anomaly_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, trans_type, float(amount), category, payment_method, str(date), description.strip(), int(is_anomaly), float(anomaly_score), anomaly_reason))
    conn.commit()
    t_id = cursor.lastrowid
    conn.close()
    return t_id

def delete_transaction(user_id, transaction_id):
    """Deletes transaction ensuring user ownership."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM transactions WHERE id = ? AND user_id = ?", (transaction_id, user_id))
    conn.commit()
    conn.close()

def get_user_transactions(user_id, limit=None, offset=0, category=None, trans_type=None, search=None, only_anomalies=False):
    """Fetches filtered transactions for the user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = "SELECT * FROM transactions WHERE user_id = ?"
    params = [user_id]
    
    if trans_type:
        query += " AND type = ?"
        params.append(trans_type)
    if category and category != 'All':
        query += " AND category = ?"
        params.append(category)
    if only_anomalies:
        query += " AND is_anomaly = 1"
    if search:
        query += " AND (description LIKE ? OR payment_method LIKE ?)"
        params.extend([f"%{search}%", f"%{search}%"])
        
    query += " ORDER BY date DESC, id DESC"
    
    if limit is not None:
        query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_transaction_counts(user_id, category=None, trans_type=None, search=None, only_anomalies=False):
    """Count matching transactions for pagination."""
    conn = get_db_connection()
    cursor = conn.cursor()
    query = "SELECT COUNT(*) as cnt FROM transactions WHERE user_id = ?"
    params = [user_id]
    if trans_type:
        query += " AND type = ?"
        params.append(trans_type)
    if category and category != 'All':
        query += " AND category = ?"
        params.append(category)
    if only_anomalies:
        query += " AND is_anomaly = 1"
    if search:
        query += " AND (description LIKE ? OR payment_method LIKE ?)"
        params.extend([f"%{search}%", f"%{search}%"])
    cursor.execute(query, params)
    cnt = cursor.fetchone()['cnt']
    conn.close()
    return cnt

def get_user_category_statistics(user_id):
    """Computes mean and std of expense amounts per category for the specific user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT category, COUNT(*) as cnt, AVG(amount) as avg_amt
        FROM transactions
        WHERE user_id = ? AND type = 'expense'
        GROUP BY category
    """, (user_id,))
    rows = cursor.fetchall()
    
    stats = {}
    for r in rows:
        cat = r['category']
        count = r['cnt']
        mean = r['avg_amt']
        
        # Calculate sample standard deviation
        cursor.execute("""
            SELECT amount FROM transactions
            WHERE user_id = ? AND type = 'expense' AND category = ?
        """, (user_id, cat))
        amounts = [x['amount'] for x in cursor.fetchall()]
        std = (sum((a - mean) ** 2 for a in amounts) / max(1, count - 1)) ** 0.5 if count > 1 else (mean * 0.4)
        
        stats[cat] = {
            'count': count,
            'mean': round(float(mean), 2),
            'std': round(float(std), 2)
        }
        
    conn.close()
    return stats

# ================= FINANCIAL AGGREGATIONS =================

def get_financial_summary(user_id):
    """
    Returns total income, total expense, net savings, savings rate,
    and current month vs previous month totals.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    now = datetime.datetime.now()
    curr_month_str = now.strftime('%Y-%m')
    prev_month_date = (now.replace(day=1) - datetime.timedelta(days=1))
    prev_month_str = prev_month_date.strftime('%Y-%m')
    
    # Lifetime totals
    cursor.execute("SELECT SUM(amount) as s FROM transactions WHERE user_id = ? AND type = 'income'", (user_id,))
    total_income = cursor.fetchone()['s'] or 0.0
    
    cursor.execute("SELECT SUM(amount) as s FROM transactions WHERE user_id = ? AND type = 'expense'", (user_id,))
    total_expense = cursor.fetchone()['s'] or 0.0
    
    # Current month
    cursor.execute("SELECT SUM(amount) as s FROM transactions WHERE user_id = ? AND type = 'income' AND date LIKE ?", (user_id, f"{curr_month_str}%"))
    curr_income = cursor.fetchone()['s'] or 0.0
    
    cursor.execute("SELECT SUM(amount) as s FROM transactions WHERE user_id = ? AND type = 'expense' AND date LIKE ?", (user_id, f"{curr_month_str}%"))
    curr_expense = cursor.fetchone()['s'] or 0.0
    
    # Previous month
    cursor.execute("SELECT SUM(amount) as s FROM transactions WHERE user_id = ? AND type = 'income' AND date LIKE ?", (user_id, f"{prev_month_str}%"))
    prev_income = cursor.fetchone()['s'] or 0.0
    
    cursor.execute("SELECT SUM(amount) as s FROM transactions WHERE user_id = ? AND type = 'expense' AND date LIKE ?", (user_id, f"{prev_month_str}%"))
    prev_expense = cursor.fetchone()['s'] or 0.0
    
    # Total Anomalies count
    cursor.execute("SELECT COUNT(*) as c FROM transactions WHERE user_id = ? AND is_anomaly = 1", (user_id,))
    anomaly_count = cursor.fetchone()['c'] or 0
    
    conn.close()
    
    net_savings = total_income - total_expense
    savings_rate = (net_savings / total_income * 100) if total_income > 0 else 0.0
    
    return {
        'total_income': round(total_income, 2),
        'total_expense': round(total_expense, 2),
        'net_savings': round(net_savings, 2),
        'savings_rate': round(savings_rate, 1),
        'curr_month_income': round(curr_income, 2),
        'curr_month_expense': round(curr_expense, 2),
        'curr_month_savings': round(curr_income - curr_expense, 2),
        'prev_month_income': round(prev_income, 2),
        'prev_month_expense': round(prev_expense, 2),
        'anomaly_count': anomaly_count,
        'current_month_name': now.strftime('%B %Y'),
        'prev_month_name': prev_month_date.strftime('%B %Y')
    }

def get_category_breakdown(user_id, month_filter=None):
    """Returns category-wise total expenses."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if month_filter:
        query = """
            SELECT category, SUM(amount) as total
            FROM transactions
            WHERE user_id = ? AND type = 'expense' AND date LIKE ?
            GROUP BY category
            ORDER BY total DESC
        """
        cursor.execute(query, (user_id, f"{month_filter}%"))
    else:
        query = """
            SELECT category, SUM(amount) as total
            FROM transactions
            WHERE user_id = ? AND type = 'expense'
            GROUP BY category
            ORDER BY total DESC
        """
        cursor.execute(query, (user_id,))
        
    rows = cursor.fetchall()
    conn.close()
    
    totals = {c: 0.0 for c in Config.CATEGORIES}
    for r in rows:
        totals[r['category']] = round(float(r['total']), 2)
        
    return totals

def get_monthly_history(user_id, months_limit=12):
    """
    Returns monthly aggregated income and expenses for charts and regression input.
    Ordered chronologically (oldest to newest).
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            SUBSTR(date, 1, 7) as year_month,
            SUM(CASE WHEN type = 'income' THEN amount ELSE 0 END) as income,
            SUM(CASE WHEN type = 'expense' THEN amount ELSE 0 END) as expense,
            SUM(CASE WHEN type = 'expense' AND category IN ('Food', 'Bills', 'Healthcare') THEN amount ELSE 0 END) as essential_expense
        FROM transactions
        WHERE user_id = ?
        GROUP BY year_month
        ORDER BY year_month ASC
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()
    
    result = []
    for r in rows:
        result.append({
            'year_month': r['year_month'],
            'income': round(float(r['income']), 2),
            'expense': round(float(r['expense']), 2),
            'savings': round(float(r['income']) - float(r['expense']), 2),
            'essential_expense': round(float(r['essential_expense']), 2)
        })
    return result[-months_limit:]

# ================= SAVINGS GOALS =================

def create_savings_goal(user_id, title, target_amount, current_amount=0.0, target_date=None, category='General Savings'):
    """Creates a new savings goal."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO savings_goals (user_id, title, target_amount, current_amount, target_date, category)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user_id, title.strip(), float(target_amount), float(current_amount), target_date, category))
    conn.commit()
    g_id = cursor.lastrowid
    conn.close()
    return g_id

def get_savings_goals(user_id):
    """Fetches all savings goals for the user with calculated progress."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM savings_goals WHERE user_id = ? ORDER BY status ASC, id DESC", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    
    goals = []
    for r in rows:
        g = dict(r)
        target = float(g['target_amount'])
        current = float(g['current_amount'])
        progress = (current / target * 100) if target > 0 else 0.0
        g['progress_pct'] = min(100.0, round(progress, 1))
        g['remaining_amount'] = max(0.0, round(target - current, 2))
        goals.append(g)
    return goals

def update_goal_progress(user_id, goal_id, added_amount):
    """Adds savings amount to an existing goal."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE savings_goals
        SET current_amount = current_amount + ?,
            status = CASE WHEN (current_amount + ?) >= target_amount THEN 'completed' ELSE 'active' END
        WHERE id = ? AND user_id = ?
    """, (float(added_amount), float(added_amount), goal_id, user_id))
    conn.commit()
    conn.close()

def delete_savings_goal(user_id, goal_id):
    """Deletes savings goal."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM savings_goals WHERE id = ? AND user_id = ?", (goal_id, user_id))
    conn.commit()
    conn.close()

import os
import sys
import json
import datetime
from functools import wraps

from flask import (
    Flask, render_template, request, redirect, 
    url_for, flash, session, jsonify, Response
)

from config import Config
from database.db import (
    init_db, create_user, authenticate_user, get_user_by_id, 
    update_user_profile, add_transaction, delete_transaction, 
    get_user_transactions, get_transaction_counts, 
    get_user_category_statistics, get_financial_summary, 
    get_category_breakdown, get_monthly_history, 
    create_savings_goal, get_savings_goals, update_goal_progress, 
    delete_savings_goal
)
from database.seed_demo_data import seed_demo_user
from ml.expense_classifier import ExpenseClassifier
from ml.expense_predictor import ExpensePredictor
from ml.anomaly_detector import AnomalyDetector
from ml.budget_advisor import BudgetAdvisor
from ml.train_all import train_all_models

app = Flask(__name__)
app.config.from_object(Config)

# Ensure folders and database initialized
init_db()

# Initialize ML Models in memory
classifier = ExpenseClassifier()
predictor = ExpensePredictor()
anomaly_detector = AnomalyDetector()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.context_processor
def inject_global_data():
    """Injects user information and global constants into all templates."""
    user = None
    if 'user_id' in session:
        user = get_user_by_id(session['user_id'])
    return {
        'current_user': user,
        'CATEGORIES': Config.CATEGORIES,
        'PAYMENT_METHODS': Config.PAYMENT_METHODS,
        'now_year': datetime.datetime.now().year
    }

# ================= AUTHENTICATION ROUTES =================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        
        user = authenticate_user(email, password)
        if user:
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            flash(f"Welcome back, {user['name']}!", 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        else:
            flash('Invalid email or password. Please verify your credentials.', 'danger')
            
    return render_template('login.html')

@app.route('/login/demo')
def login_demo():
    """Instant 1-click login for project examiners, professors, and testing."""
    u_id = seed_demo_user(force=False)
    user = get_user_by_id(u_id)
    session['user_id'] = user['id']
    session['user_name'] = user['name']
    flash('Logged into Demonstration Account (Alex Sharma) with 1-Year Financial History!', 'success')
    return redirect(url_for('dashboard'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        monthly_income = request.form.get('monthly_income', 50000.0)
        
        if not name or not email or not password:
            flash('All required fields must be filled.', 'warning')
            return render_template('register.html')
            
        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'warning')
            return render_template('register.html')
            
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('register.html')
            
        try:
            inc = float(monthly_income)
        except ValueError:
            inc = 50000.0
            
        user_id, err = create_user(name=name, email=email, password=password, monthly_income=inc)
        if err:
            flash(err, 'danger')
        else:
            session['user_id'] = user_id
            session['user_name'] = name
            flash('Account created successfully! Welcome to your AI Personal Finance Advisor.', 'success')
            return redirect(url_for('dashboard'))
            
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out securely.', 'info')
    return redirect(url_for('login'))

# ================= CORE PAGES =================

@app.route('/')
@app.route('/dashboard')
@login_required
def dashboard():
    u_id = session['user_id']
    user = get_user_by_id(u_id)
    summary = get_financial_summary(u_id)
    recent_txns = get_user_transactions(u_id, limit=8)
    goals = get_savings_goals(u_id)
    monthly_hist = get_monthly_history(u_id, months_limit=12)
    cat_totals = get_category_breakdown(u_id)
    
    # ML Forecast for Next Month
    pred_res = predictor.predict_next_month(monthly_hist, current_income=user['monthly_income'])
    
    # Highest Spending Category
    highest_cat = None
    highest_amt = 0.0
    for cat, amt in cat_totals.items():
        if amt > highest_amt:
            highest_amt = amt
            highest_cat = cat
            
    # AI Financial Insights
    curr_data = {'total_expense': summary['curr_month_expense'], 'categories': get_category_breakdown(u_id, datetime.datetime.now().strftime('%Y-%m'))}
    prev_date = (datetime.datetime.now().replace(day=1) - datetime.timedelta(days=1))
    prev_data = {'total_expense': summary['prev_month_expense'], 'categories': get_category_breakdown(u_id, prev_date.strftime('%Y-%m'))}
    ai_insights = BudgetAdvisor.generate_insights(curr_data, prev_data, user['monthly_income'])
    
    return render_template(
        'dashboard.html',
        summary=summary,
        recent_txns=recent_txns,
        goals=goals,
        prediction=pred_res,
        highest_cat=highest_cat,
        highest_amt=highest_amt,
        ai_insights=ai_insights
    )

@app.route('/transactions')
@login_required
def transactions():
    u_id = session['user_id']
    page = request.args.get('page', 1, type=int)
    per_page = 15
    offset = (page - 1) * per_page
    
    cat_filter = request.args.get('category', 'All')
    type_filter = request.args.get('type', '')
    search = request.args.get('search', '').strip()
    only_anom = request.args.get('anomalies', '') == '1'
    
    total_records = get_transaction_counts(
        u_id, 
        category=cat_filter, 
        trans_type=type_filter, 
        search=search, 
        only_anomalies=only_anom
    )
    
    txns = get_user_transactions(
        u_id,
        limit=per_page,
        offset=offset,
        category=cat_filter,
        trans_type=type_filter,
        search=search,
        only_anomalies=only_anom
    )
    
    total_pages = max(1, (total_records + per_page - 1) // per_page)
    
    return render_template(
        'transactions.html',
        transactions=txns,
        page=page,
        total_pages=total_pages,
        total_records=total_records,
        current_cat=cat_filter,
        current_type=type_filter,
        search=search,
        only_anomalies=only_anom
    )

@app.route('/add-transaction', methods=['GET', 'POST'])
@login_required
def add_new_transaction():
    u_id = session['user_id']
    if request.method == 'POST':
        trans_type = request.form.get('type', 'expense')
        amount_raw = request.form.get('amount', 0)
        category = request.form.get('category', '').strip()
        payment_method = request.form.get('payment_method', 'Credit Card')
        date = request.form.get('date', datetime.date.today().strftime('%Y-%m-%d'))
        description = request.form.get('description', '').strip()
        
        try:
            amount = float(amount_raw)
            if amount <= 0:
                raise ValueError()
        except ValueError:
            flash('Please enter a valid positive numerical amount.', 'danger')
            return render_template('add_transaction.html')
            
        if not description:
            flash('Please provide a description for the transaction.', 'warning')
            return render_template('add_transaction.html')
            
        # ML Categorization if selected as Auto-Detect or blank
        if not category or category == 'Auto':
            predicted_cat, conf = classifier.predict(description)
            category = predicted_cat
            flash(f"AI Categorized this transaction as '{category}' ({conf*100:.1f}% confidence)", 'info')
            
        # ML Anomaly Detection (only for expenses)
        is_anom = 0
        score = 0.0
        reason = None
        if trans_type == 'expense':
            cat_stats = get_user_category_statistics(u_id)
            anom_res = anomaly_detector.detect(amount, category, cat_stats)
            if anom_res['is_anomaly']:
                is_anom = 1
                score = anom_res['anomaly_score']
                reason = anom_res['warning']
                flash(f"⚠️ Unusual Spending Detected: {reason}", 'warning')
                
        add_transaction(
            user_id=u_id,
            trans_type=trans_type,
            amount=amount,
            category=category,
            payment_method=payment_method,
            date=date,
            description=description,
            is_anomaly=is_anom,
            anomaly_score=score,
            anomaly_reason=reason
        )
        flash('Transaction recorded successfully!', 'success')
        return redirect(url_for('transactions'))
        
    return render_template('add_transaction.html', today=datetime.date.today().strftime('%Y-%m-%d'))

@app.route('/delete-transaction/<int:trans_id>', methods=['POST'])
@login_required
def remove_transaction(trans_id):
    u_id = session['user_id']
    delete_transaction(u_id, trans_id)
    flash('Transaction deleted successfully.', 'info')
    return redirect(request.referrer or url_for('transactions'))

# ================= AI & ML PAGES =================

@app.route('/expense-prediction')
@login_required
def expense_prediction():
    u_id = session['user_id']
    user = get_user_by_id(u_id)
    monthly_hist = get_monthly_history(u_id, months_limit=12)
    prediction = predictor.predict_next_month(monthly_hist, current_income=user['monthly_income'])
    
    # Load trained model metrics
    metrics = {}
    if os.path.exists(Config.METRICS_PATH):
        try:
            with open(Config.METRICS_PATH, 'r') as f:
                data = json.load(f)
                metrics = data.get('models', {}).get('expense_predictor', {})
        except Exception:
            metrics = {}
            
    return render_template(
        'expense_prediction.html',
        prediction=prediction,
        monthly_history=monthly_hist,
        metrics=metrics,
        user=user
    )

@app.route('/budget-recommendation')
@login_required
def budget_recommendation():
    u_id = session['user_id']
    user = get_user_by_id(u_id)
    cat_totals = get_category_breakdown(u_id, datetime.datetime.now().strftime('%Y-%m'))
    goals = get_savings_goals(u_id)
    budget_data = BudgetAdvisor.generate_recommendations(
        monthly_income=user['monthly_income'],
        category_totals=cat_totals,
        active_goals=goals
    )
    return render_template('budget_recommendation.html', budget=budget_data, user=user)

@app.route('/ai-insights')
@login_required
def ai_insights_page():
    u_id = session['user_id']
    user = get_user_by_id(u_id)
    summary = get_financial_summary(u_id)
    cat_totals = get_category_breakdown(u_id)
    
    curr_data = {'total_expense': summary['curr_month_expense'], 'categories': get_category_breakdown(u_id, datetime.datetime.now().strftime('%Y-%m'))}
    prev_date = (datetime.datetime.now().replace(day=1) - datetime.timedelta(days=1))
    prev_data = {'total_expense': summary['prev_month_expense'], 'categories': get_category_breakdown(u_id, prev_date.strftime('%Y-%m'))}
    
    insights = BudgetAdvisor.generate_insights(curr_data, prev_data, user['monthly_income'])
    anomalies = get_user_transactions(u_id, only_anomalies=True, limit=10)
    
    return render_template(
        'ai_insights.html',
        insights=insights,
        anomalies=anomalies,
        summary=summary,
        category_totals=cat_totals,
        user=user
    )

@app.route('/savings-goals', methods=['GET', 'POST'])
@login_required
def savings_goals_page():
    u_id = session['user_id']
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'create':
            title = request.form.get('title', '').strip()
            target_amount = float(request.form.get('target_amount', 0))
            current_amount = float(request.form.get('current_amount', 0))
            target_date = request.form.get('target_date', '')
            category = request.form.get('category', 'General Savings')
            if title and target_amount > 0:
                create_savings_goal(u_id, title, target_amount, current_amount, target_date, category)
                flash(f"Savings goal '{title}' created successfully!", 'success')
            else:
                flash('Please enter a valid title and target amount.', 'danger')
        elif action == 'deposit':
            goal_id = int(request.form.get('goal_id'))
            add_amt = float(request.form.get('amount', 0))
            if add_amt > 0:
                update_goal_progress(u_id, goal_id, add_amt)
                flash(f"Deposited ₹{add_amt:,.2f} towards your goal!", 'success')
        elif action == 'delete':
            goal_id = int(request.form.get('goal_id'))
            delete_savings_goal(u_id, goal_id)
            flash('Savings goal deleted.', 'info')
            
        return redirect(url_for('savings_goals_page'))
        
    goals = get_savings_goals(u_id)
    return render_template('savings_goals.html', goals=goals)

@app.route('/ml-models')
@login_required
def ml_models_page():
    """Dedicated Viva & Presentation showcase of all AIML components."""
    metrics_data = {}
    if os.path.exists(Config.METRICS_PATH):
        try:
            with open(Config.METRICS_PATH, 'r') as f:
                metrics_data = json.load(f)
        except Exception:
            metrics_data = {}
            
    return render_template('ml_models.html', metrics=metrics_data)

@app.route('/reports')
@login_required
def reports_page():
    u_id = session['user_id']
    user = get_user_by_id(u_id)
    summary = get_financial_summary(u_id)
    monthly_hist = get_monthly_history(u_id, months_limit=12)
    cat_totals = get_category_breakdown(u_id)
    goals = get_savings_goals(u_id)
    anomalies = get_user_transactions(u_id, only_anomalies=True, limit=5)
    pred_res = predictor.predict_next_month(monthly_hist, current_income=user['monthly_income'])
    budget_data = BudgetAdvisor.generate_recommendations(user['monthly_income'], cat_totals, goals)
    
    return render_template(
        'reports.html',
        user=user,
        summary=summary,
        monthly_hist=monthly_hist,
        cat_totals=cat_totals,
        goals=goals,
        anomalies=anomalies,
        prediction=pred_res,
        budget=budget_data,
        report_date=datetime.date.today().strftime('%B %d, %Y')
    )

@app.route('/export/csv')
@login_required
def export_csv():
    """Generates and downloads a CSV file of all transactions."""
    u_id = session['user_id']
    txns = get_user_transactions(u_id)
    
    import io
    import csv
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Date', 'Type', 'Amount (INR)', 'Category', 'Payment Method', 'Description', 'Is Anomaly', 'Anomaly Reason'])
    for t in txns:
        writer.writerow([
            t['id'], t['date'], t['type'], t['amount'], t['category'],
            t['payment_method'], t['description'], 'YES' if t['is_anomaly'] else 'NO',
            t['anomaly_reason'] or ''
        ])
        
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=transactions_export.csv"}
    )

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile_page():
    u_id = session['user_id']
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'update_profile':
            name = request.form.get('name', '').strip()
            monthly_income = float(request.form.get('monthly_income', 50000.0))
            occupation = request.form.get('occupation', 'Student / Professional')
            update_user_profile(u_id, name, monthly_income, occupation)
            session['user_name'] = name
            flash('Profile updated successfully!', 'success')
        elif action == 'reload_demo':
            seed_demo_user(force=True)
            flash('Demo data refreshed with 12 months of rich historical records!', 'success')
        return redirect(url_for('profile_page'))
        
    user = get_user_by_id(u_id)
    summary = get_financial_summary(u_id)
    return render_template('profile.html', user=user, summary=summary)

# ================= REST / AJAX API ENDPOINTS =================

@app.route('/api/predict-category', methods=['GET'])
@login_required
def api_predict_category():
    """Real-time AJAX endpoint for auto-categorization preview."""
    desc = request.args.get('desc', '').strip()
    if not desc:
        return jsonify({'category': 'Others', 'confidence': 0.0, 'probabilities': {}})
    category, conf = classifier.predict(desc)
    all_probs = classifier.predict_all_probabilities(desc)
    return jsonify({
        'category': category,
        'confidence': round(conf * 100, 1),
        'probabilities': all_probs
    })

@app.route('/api/check-anomaly', methods=['POST'])
@login_required
def api_check_anomaly():
    """Asynchronous anomaly check before transaction submit."""
    u_id = session['user_id']
    data = request.get_json() or {}
    amount = float(data.get('amount', 0))
    category = data.get('category', 'Others')
    
    cat_stats = get_user_category_statistics(u_id)
    res = anomaly_detector.detect(amount, category, cat_stats)
    return jsonify(res)

@app.route('/api/chart-data/overview')
@login_required
def api_chart_overview():
    """Chart.js monthly trend data."""
    u_id = session['user_id']
    history = get_monthly_history(u_id, months_limit=12)
    labels = [h['year_month'] for h in history]
    incomes = [h['income'] for h in history]
    expenses = [h['expense'] for h in history]
    savings = [h['savings'] for h in history]
    return jsonify({
        'labels': labels,
        'income': incomes,
        'expense': expenses,
        'savings': savings
    })

@app.route('/api/chart-data/category')
@login_required
def api_chart_category():
    """Chart.js category doughnut data."""
    u_id = session['user_id']
    month = request.args.get('month')
    cat_totals = get_category_breakdown(u_id, month_filter=month)
    # Filter out categories with 0 spend
    labels = []
    data = []
    for cat, val in cat_totals.items():
        if val > 0:
            labels.append(cat)
            data.append(val)
    return jsonify({
        'labels': labels,
        'data': data
    })

@app.route('/api/retrain-models', methods=['POST'])
@login_required
def api_retrain_models():
    """Retrain all ML models live from the ML dashboard page."""
    try:
        new_metrics = train_all_models()
        # Reload models in memory
        classifier.load_model()
        predictor.load_model()
        anomaly_detector.load_model()
        return jsonify({'status': 'success', 'message': 'Models retrained successfully!', 'metrics': new_metrics})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    # Default development server
    port = int(os.environ.get('PORT', 5000))
    print(f"\n[AI-Powered Personal Finance Advisor] Starting server on http://127.0.0.1:{port}")
    app.run(host='127.0.0.1', port=port, debug=False)

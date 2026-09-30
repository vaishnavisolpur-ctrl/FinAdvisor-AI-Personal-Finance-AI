import os
import random
import csv
import pandas as pd
import numpy as np

# Ensure datasets directory exists
DATASETS_DIR = os.path.abspath(os.path.dirname(__file__))
os.makedirs(DATASETS_DIR, exist_ok=True)

# 1. EXPENSE CATEGORIZATION DATASET GENERATION
# Realistic templates and merchants per category
CATEGORY_TEMPLATES = {
    'Food': [
        "Swiggy order from {rest}", "Zomato delivery from {rest}", "Lunch at {rest}", "Dinner with team at {rest}",
        "Starbucks {coffee}", "Cafe Coffee Day {coffee}", "Costa Coffee {coffee}", "Blue Tokai {coffee}",
        "Grocery purchase at {supermarket}", "Daily fruits and vegetables from {supermarket}",
        "Milk, eggs, and bread delivery from {grocery_app}", "Blinkit grocery order", "Zepto quick delivery",
        "Instamart snacks and dairy", "Subway sub and cookies", "McDonald's burger combo meal",
        "KFC chicken bucket meal", "Domino's cheese burst pizza order", "Pizza Hut dinner feast",
        "Haldiram's sweets and thali", "Chai Point ginger tea snacks", "Bakery fresh artisan bread & pastry",
        "Supermarket weekly provisions", "Local meat market fresh fish", "Organic farmers market vegetables",
        "Juice bar fresh fruit juice", "Chaat stall evening snacks", "Dunkin Donuts box of 6",
        "Ice cream parlour Baskin Robbins", "Biryani pot meal box Behrouz"
    ],
    'Transport': [
        "Uber cab ride to {dest}", "Ola auto ride to {dest}", "Rapido bike taxi ride",
        "Metro smart card recharge", "Delhi Metro token recharge", "Bangalore Namma Metro recharge",
        "Petrol refill at Shell station", "BPCL petrol pump fuel car", "HP petrol bunk diesel refill",
        "Indian Oil fuel payment", "FASTag highway toll recharge", "Airport parking ticket fee",
        "Railway train ticket booking IRCTC", "Flight ticket booking IndiGo {city}", "Air India flight to {city}",
        "redBus interstate bus booking", "Auto rickshaw cash fare", "Car servicing and oil change at Maruti",
        "Hyundai annual car maintenance", "Honda two wheeler bike service", "Bicycle repair and tire puncture",
        "Airport express metro ticket", "Monthly city bus pass pass renewal", "Ferry boat ticket ride",
        "EV vehicle fast charging station fee"
    ],
    'Shopping': [
        "Amazon India online electronics order", "Flipkart Big Billion Days purchase", "Myntra fashion summer clothing",
        "Zara formal shirt and trousers", "H&M slim fit cotton t-shirt", "Uniqlo heattech jacket",
        "Nike running shoes Air Zoom", "Adidas training sportswear", "Puma casual sneakers sale",
        "Decathlon badminton racket and kit", "IKEA furniture study desk chair", "Home Centre curtains and cushions",
        "Lenskart computer eyeglasses and frame", "Titan quartz wrist watch showroom", "Nykaa cosmetics skincare serum",
        "Sephora perfume and makeup", "Apple Store iPhone case & magsafe", "Croma electronics laptop bag",
        "Reliance Digital wireless earbuds", "Westside lifestyle casual wear", "Pantaloons jeans and belt",
        "Fabindia ethnic kurta collection", "Shoppers Stop leather wallet", "Bookstore fiction paperback novel"
    ],
    'Bills': [
        "Electricity bill payment BESCOM", "Tata Power residential electricity bill", "Adani Electricity monthly bill",
        "Piped natural gas IGL monthly bill", "Adani Gas cylinder booking", "Municipal Corporation water utility bill",
        "Airtel Xstream broadband internet bill", "JioFiber high speed optical fiber bill", "ACT Fibernet broadband bill",
        "Jio mobile postpaid monthly plan", "Airtel mobile postpaid plan recharge", "Vi Vodafone Idea monthly bill",
        "Apartment maintenance society dues", "House rent transfer via NEFT", "DTH Tata Play TV recharge",
        "Airtel Digital TV subscription recharge", "Municipal property tax annual payment", "HDFC Credit card statement payment",
        "ICICI Bank credit card bill payment", "SBI Card monthly settlement", "Waste management & recycling collection fee",
        "Building security and upkeep fund"
    ],
    'Education': [
        "Coursera Machine Learning Specialization subscription", "Udemy Full-Stack Web Development course",
        "University semester academic tuition fee", "College exam fee online portal",
        "DataCamp Data Science career track pass", "EdX MIT micro-masters certificate fee",
        "College library annual membership fee", "Technical textbooks bookstore purchase",
        "Stationery supplies notebook pens scientific calculator", "GRE test official registration ETS",
        "IELTS academic English exam fee", "TOEFL test registration payment", "AWS Cloud Certified Architect exam voucher",
        "LeetCode annual premium subscription", "Coding Ninjas competitive programming course",
        "Pluralsight annual tech skills subscription", "Masterclass annual streaming pass",
        "Research paper IEEE publication fee", "School quarterly bus and tuition fee", "Tutoring center monthly coaching fee"
    ],
    'Entertainment': [
        "Netflix monthly 4K UHD streaming subscription", "Amazon Prime Video annual renewal",
        "Disney+ Hotstar Super annual plan", "Spotify Premium Family music subscription",
        "Apple Music monthly individual subscription", "YouTube Premium student subscription",
        "BookMyShow movie tickets PVR IMAX 3D", "PVR Cinemas caramel popcorn and beverage",
        "INOX multiplex movie ticket booking", "Steam store winter gaming sale PC game",
        "PlayStation Network Plus game subscription", "Nintendo Switch digital eShop game",
        "Gaming arcade gaming card recharge", "Live standup comedy show ticket BMS",
        "Music concert festival pass", "Amusement park theme park entry pass",
        "Bowling alley weekend game with friends", "Board game club entry fee",
        "Audible audiobooks monthly credit", "Kindle Unlimited reading subscription"
    ],
    'Healthcare': [
        "Apollo Pharmacy prescription medicine", "Medplus medicines and first aid", "1mg online order health supplements",
        "Netmeds pharmacy home delivery", "Doctor consultation fee general physician",
        "Dental cleaning and cavity filling clinic", "Eye checkup and optometrist consultation",
        "Thyrocare full body health checkup package", "Dr Lal PathLabs blood lipid profile test",
        "Diagnostic radiology X-ray scan report", "Prescription multivitamins and Omega 3",
        "Star Health Insurance annual premium", "HDFC ERGO Mediclaim health policy payment",
        "Physiotherapy rehabilitation therapy session", "Ayurvedic clinic consultation & herbs",
        "Dermatologist skin treatment & ointment", "Orthopedic knee support brace",
        "Pediatrician child vaccination shot fee", "Hospital emergency room care charges",
        "Mental wellness therapist counseling session"
    ],
    'Others': [
        "ATM cash withdrawal cash dispensing", "Donation to PM Relief Fund / UNICEF",
        "Pet clinic veterinarian checkup vaccination", "Pet store premium dog food kibble",
        "Speed Post courier parcel shipping charges", "Blue Dart express courier delivery",
        "Dry cleaning laundry suit & blazer service", "Key maker locksmith urgent home visit",
        "Carpentry repair work service handyman", "Plumber leakage repair bathroom fitting",
        "Annual bank locker rent charge debit", "Bank debit card annual maintenance charge",
        "Flower bouquet anniversary surprise delivery", "Gifting greeting card and luxury gift box",
        "Tailor alteration stitching charges suit", "Passport seva online application fee",
        "Driving license renewal RTO fee", "Notary public stamp paper affidavit fee",
        "Charity donation local animal shelter", "Car wash and detailing spa service"
    ]
}

# Fillers for templates
FILLERS = {
    'rest': ['Punjabi Rasoi', 'Barbeque Nation', 'Saravana Bhavan', 'Truffles', 'Meghana Foods', 'Toit Brewery', 'Olive Bistro', 'Mainland China', 'Empire Restaurant', 'Nandhini Deluxe', 'Chowman', 'Smoke House Deli', 'Pizza Bakery', 'Bawarchi Biryani', 'Cafe Mocha'],
    'coffee': ['Caramel Macchiato', 'Hazelnut Cold Brew', 'Cappuccino Tall', 'Espresso Double Shot', 'Classic Iced Latte', 'Americano Hot', 'Mocha Frappuccino', 'Vanilla Sweet Cream'],
    'supermarket': ['Nature\'s Basket', 'Big Bazaar', 'Reliance Smart', 'More Megastore', 'Spencers Supermarket', 'Star Bazaar', 'Metro Cash & Carry', 'D-Mart Supermarket'],
    'grocery_app': ['Country Delight', 'Otpy Superstore', 'Milkbasket', 'Dunzo Daily', 'FreshToHome'],
    'dest': ['Airport Terminal 2', 'Railway Central Station', 'Tech Park Electronic City', 'Downtown Office Hub', 'Home Sweet Home', 'Shopping Promenade Mall', 'Medical Center Hospital', 'University Campus Gate'],
    'city': ['Mumbai BOM', 'Delhi DEL', 'Bengaluru BLR', 'Hyderabad HYD', 'Chennai MAA', 'Kolkata CCU', 'Goa GOI', 'Pune PNQ']
}

def generate_classification_dataset(target_rows=2400):
    rows = []
    categories = list(CATEGORY_TEMPLATES.keys())
    
    # Generate balanced samples
    per_cat = target_rows // len(categories)
    
    for cat in categories:
        templates = CATEGORY_TEMPLATES[cat]
        for _ in range(per_cat):
            tmpl = random.choice(templates)
            # fill placeholders
            desc = tmpl
            for key, val_list in FILLERS.items():
                if f"{{{key}}}" in desc:
                    desc = desc.replace(f"{{{key}}}", random.choice(val_list))
            
            # Add occasional realistic variations (random punctuation, numbers, lower/upper case)
            var_type = random.random()
            if var_type < 0.15:
                desc = desc.lower()
            elif var_type < 0.25:
                desc = desc.upper()
            elif var_type < 0.35:
                desc = f"{desc} - Ref #{random.randint(1000, 99999)}"
            elif var_type < 0.45:
                desc = f"POS txn: {desc}"
            
            rows.append({
                'description': desc,
                'category': cat
            })
            
    # Shuffle
    random.seed(42)
    random.shuffle(rows)
    
    csv_path = os.path.join(DATASETS_DIR, 'expense_categories_dataset.csv')
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(df)} classification records at: {csv_path}")
    return df

# 2. HISTORICAL MONTHLY EXPENSES FOR REGRESSION MODEL
def generate_monthly_regression_dataset(num_records=1200):
    """
    Generates synthetic monthly spending profiles across multiple synthetic user profiles
    with realistic correlations:
    - lag_1_expense: prior month spending
    - lag_2_expense: 2-months prior spending
    - rolling_3m_mean: moving average
    - monthly_income: salary / income
    - essential_ratio: proportion of food + bills + healthcare (40-75%)
    - month: captures seasonality (e.g. holiday peaks in Oct/Nov/Dec or summer)
    - actual_expense: target next-month expense (correlated with lags + income + noise)
    """
    np.random.seed(42)
    records = []
    
    # Generate for 60 synthetic users across 20-30 months each
    user_bases = [
        {'income': 35000, 'base_expense': 24000, 'volatility': 2200},
        {'income': 50000, 'base_expense': 32000, 'volatility': 2800},
        {'income': 75000, 'base_expense': 46000, 'volatility': 4000},
        {'income': 100000, 'base_expense': 58000, 'volatility': 5500},
        {'income': 140000, 'base_expense': 75000, 'volatility': 7000},
        {'income': 180000, 'base_expense': 92000, 'volatility': 9000},
    ]
    
    for user_id in range(1, 61):
        profile = random.choice(user_bases)
        base_inc = profile['income'] * random.uniform(0.9, 1.15)
        base_exp = profile['base_expense'] * random.uniform(0.88, 1.12)
        vol = profile['volatility']
        
        # 24 consecutive months
        prev_1 = base_exp + np.random.normal(0, vol * 0.7)
        prev_2 = base_exp + np.random.normal(0, vol * 0.7)
        prev_3 = base_exp + np.random.normal(0, vol * 0.7)
        
        for m_idx in range(1, 25):
            month = ((m_idx - 1) % 12) + 1
            # Seasonality: Nov/Dec holiday spike (+15%), Feb dip (-8%)
            seasonal_factor = 1.0
            if month in [10, 11, 12]:
                seasonal_factor = 1.12
            elif month == 2:
                seasonal_factor = 0.94
                
            rolling_3m = np.mean([prev_1, prev_2, prev_3])
            rolling_std = np.std([prev_1, prev_2, prev_3]) + 100.0
            
            essential_pct = random.uniform(0.42, 0.68)
            discretionary_pct = 1.0 - essential_pct
            savings_rate = max(0.05, 1.0 - (base_exp / base_inc))
            savings_target = base_inc * savings_rate
            
            # Target expense for this month: 
            # 50% lag_1 + 30% rolling_3m + 15% income-adjustment + seasonality + random noise
            exp_target = (
                0.48 * prev_1 + 
                0.28 * rolling_3m + 
                0.12 * (base_inc * (1 - savings_rate)) + 
                np.random.normal(0, vol * 0.6)
            ) * seasonal_factor
            
            exp_target = max(base_exp * 0.5, min(base_inc * 1.1, exp_target))
            
            records.append({
                'user_id': user_id,
                'month_idx': m_idx,
                'month_of_year': month,
                'monthly_income': round(base_inc, 2),
                'lag_1_expense': round(prev_1, 2),
                'lag_2_expense': round(prev_2, 2),
                'lag_3_expense': round(prev_3, 2),
                'rolling_3m_mean': round(rolling_3m, 2),
                'rolling_3m_std': round(rolling_std, 2),
                'essential_ratio': round(essential_pct, 4),
                'discretionary_ratio': round(discretionary_pct, 4),
                'savings_target': round(savings_target, 2),
                'actual_expense': round(exp_target, 2)
            })
            
            # shift lags
            prev_3 = prev_2
            prev_2 = prev_1
            prev_1 = exp_target
            
    csv_path = os.path.join(DATASETS_DIR, 'historical_monthly_expenses.csv')
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(df)} monthly regression records at: {csv_path}")
    return df

# 3. SAMPLE DEMO TRANSACTIONS FOR SEEDING
def generate_sample_transactions_csv():
    # Will be combined with realistic dates, descriptions, categories, payment methods
    records = []
    # Generates a baseline CSV for demonstration
    csv_path = os.path.join(DATASETS_DIR, 'transactions_sample.csv')
    # Will be seeded cleanly into SQLite as well
    pass

if __name__ == '__main__':
    generate_classification_dataset(2400)
    generate_monthly_regression_dataset(1440)
    print("All datasets generated successfully!")

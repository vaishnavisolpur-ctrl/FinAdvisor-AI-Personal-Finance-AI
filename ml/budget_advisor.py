import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config

class BudgetAdvisor:
    """
    Intelligent Personal Budget Recommendation & Financial Insights Engine.
    
    Combines empirical finance principles (50/30/20 rule: Needs, Wants, Savings)
    with adaptive elasticity weights derived from the user's historical transaction behavior
    and active savings goals.
    """
    
    DISCLAIMER = (
        "Disclaimer: These budget allocations and financial insights are data-driven algorithmic guidelines "
        "calculated from historical spending patterns and mathematical optimization models. "
        "They are intended for informational and educational purposes only and do not constitute certified professional financial advice."
    )
    
    # Category Classification into Needs vs Wants
    CATEGORY_TYPES = {
        'Food': 'Needs',            # Essential groceries + some dining
        'Bills': 'Needs',           # Utilities, rent, broadband
        'Healthcare': 'Needs',      # Medical & insurance
        'Transport': 'Needs',       # Commute & fuel
        'Education': 'Needs',       # Skill building & tuition
        'Shopping': 'Wants',        # Fashion, gadgets, lifestyle
        'Entertainment': 'Wants',   # Movies, games, streaming
        'Others': 'Wants'           # Miscellaneous
    }
    
    @classmethod
    def generate_recommendations(cls, monthly_income: float, category_totals: dict, active_goals: list = None):
        """
        Generates personalized category budgets based on user's income, current spending, and active goals.
        """
        monthly_income = float(monthly_income) if monthly_income > 0 else 50000.0
        active_goals = active_goals or []
        
        # Total current monthly expenses
        total_spent = sum(category_totals.values()) if category_totals else 0.0
        
        # Calculate goal demand (how much user needs to save per month for goals)
        goal_monthly_demand = 0.0
        for g in active_goals:
            rem = max(0.0, g.get('target_amount', 0) - g.get('current_amount', 0))
            # Assume 6-month horizon if target date not specified
            goal_monthly_demand += (rem / 6.0)
            
        # Target 50/30/20 baseline:
        # Needs: 50% = income * 0.50
        # Wants: 30% = income * 0.30
        # Savings: 20% = income * 0.20 (or higher if goals demand)
        ideal_needs = monthly_income * 0.50
        ideal_wants = monthly_income * 0.30
        
        # Adaptive savings rate
        suggested_savings = max(monthly_income * 0.20, min(monthly_income * 0.40, goal_monthly_demand))
        remaining_for_spend = monthly_income - suggested_savings
        
        # Category weight allocation
        # Baseline ideal weights within spending
        base_weights = {
            'Food': 0.25,
            'Bills': 0.25,
            'Transport': 0.12,
            'Healthcare': 0.08,
            'Education': 0.08,
            'Shopping': 0.10,
            'Entertainment': 0.07,
            'Others': 0.05
        }
        
        # Adapt weights if user has history
        adapted_weights = {}
        if total_spent > 0:
            for cat in Config.CATEGORIES:
                user_share = category_totals.get(cat, 0.0) / total_spent
                # Blend 60% historical reality + 40% ideal financial discipline
                adapted_weights[cat] = 0.60 * user_share + 0.40 * base_weights.get(cat, 0.1)
            # Normalize
            w_sum = sum(adapted_weights.values())
            for cat in adapted_weights:
                adapted_weights[cat] /= w_sum
        else:
            adapted_weights = base_weights
            
        recommendations = {}
        for cat in Config.CATEGORIES:
            allocated = remaining_for_spend * adapted_weights.get(cat, 0.1)
            current = category_totals.get(cat, 0.0)
            status = 'Within Budget'
            pct_utilization = (current / allocated * 100) if allocated > 0 else 0
            if current > allocated * 1.15:
                status = 'Over Budget'
            elif current > allocated * 0.85:
                status = 'Near Limit'
                
            recommendations[cat] = {
                'category': cat,
                'type': cls.CATEGORY_TYPES.get(cat, 'Wants'),
                'recommended_budget': round(allocated, 2),
                'current_spent': round(current, 2),
                'utilization_pct': round(pct_utilization, 1),
                'status': status,
                'weight_pct': round(adapted_weights.get(cat, 0.1) * 100, 1)
            }
            
        summary = {
            'monthly_income': round(monthly_income, 2),
            'suggested_savings': round(suggested_savings, 2),
            'savings_rate_pct': round((suggested_savings / monthly_income) * 100, 1),
            'discretionary_budget': round(remaining_for_spend, 2),
            'categories': recommendations,
            'rule_50_30_20': {
                'needs_budget': round(ideal_needs, 2),
                'wants_budget': round(ideal_wants, 2),
                'savings_target': round(suggested_savings, 2)
            },
            'disclaimer': cls.DISCLAIMER
        }
        return summary

    @classmethod
    def generate_insights(cls, current_month_data: dict, prev_month_data: dict, current_income: float):
        """
        Generates actionable, data-driven AI financial insights comparing months and patterns.
        """
        insights = []
        
        curr_total = current_month_data.get('total_expense', 0.0)
        prev_total = prev_month_data.get('total_expense', 0.0)
        curr_cats = current_month_data.get('categories', {})
        prev_cats = prev_month_data.get('categories', {})
        
        # 1. Overall Month-over-Month Spending Trend
        if prev_total > 0:
            diff = curr_total - prev_total
            pct = (diff / prev_total) * 100
            if pct > 10:
                insights.append({
                    'type': 'warning',
                    'icon': 'trending-up',
                    'title': 'Overall Spending Spiked',
                    'message': f"Your overall expenses increased by {abs(pct):.1f}% (₹{abs(diff):,.2f}) compared to the previous month. Review discretionary spending."
                })
            elif pct < -10:
                insights.append({
                    'type': 'success',
                    'icon': 'trending-down',
                    'title': 'Excellent Spending Reduction',
                    'message': f"Great job! Your spending is down {abs(pct):.1f}% (₹{abs(diff):,.2f}) compared to last month."
                })
            else:
                insights.append({
                    'type': 'info',
                    'icon': 'activity',
                    'title': 'Consistent Spending Rate',
                    'message': f"Your total expenditure is holding steady within ±10% of last month's spending level."
                })
                
        # 2. Dominant Spending Category
        if curr_cats:
            top_cat, top_amt = max(curr_cats.items(), key=lambda x: x[1])
            if curr_total > 0:
                share = (top_amt / curr_total) * 100
                insights.append({
                    'type': 'info',
                    'icon': 'pie-chart',
                    'title': f'{top_cat} is Your Highest Category',
                    'message': f"{top_cat} represents {share:.1f}% (₹{top_amt:,.2f}) of your total monthly expenses so far."
                })
                
        # 3. Category Specific Month-over-Month Shifts
        for cat in ['Food', 'Shopping', 'Entertainment', 'Transport']:
            c_val = curr_cats.get(cat, 0.0)
            p_val = prev_cats.get(cat, 0.0)
            if p_val > 0 and c_val > p_val * 1.25 and (c_val - p_val) > 1000:
                pct_inc = ((c_val - p_val) / p_val) * 100
                insights.append({
                    'type': 'warning',
                    'icon': 'alert-circle',
                    'title': f'{cat} Expenses Increased',
                    'message': f"Your {cat.lower()} expenses surged by {pct_inc:.1f}% compared with the previous month."
                })
                
        # 4. Savings Rate Health
        if current_income > 0:
            savings = current_income - curr_total
            savings_rate = (savings / current_income) * 100
            if savings_rate >= 25:
                insights.append({
                    'type': 'success',
                    'icon': 'shield-check',
                    'title': 'Robust Savings Discipline',
                    'message': f"You are saving {savings_rate:.1f}% of your monthly income, exceeding the recommended 20% threshold!"
                })
            elif savings_rate < 10:
                insights.append({
                    'type': 'danger',
                    'icon': 'alert-triangle',
                    'title': 'Low Savings Margin Alert',
                    'message': "Your estimated monthly savings can improve if discretionary spending on Shopping and Entertainment is curtailed."
                })
                
        # 5. Wants vs Needs Balance
        wants_total = sum(curr_cats.get(c, 0.0) for c in ['Shopping', 'Entertainment', 'Others'])
        if curr_total > 0:
            wants_pct = (wants_total / curr_total) * 100
            if wants_pct > 40:
                insights.append({
                    'type': 'warning',
                    'icon': 'compass',
                    'title': 'High Discretionary Spending Ratio',
                    'message': f"Discretionary wants account for {wants_pct:.1f}% of your spending (recommended: ≤ 30%). Prioritizing essentials can free up surplus capital."
                })
                
        # Ensure at least 3 insights always
        if len(insights) < 3:
            insights.append({
                'type': 'info',
                'icon': 'award',
                'title': 'AI Advisory Tip',
                'message': 'Automating your savings transfer on salary day is the most reliable way to achieve your medium and long-term financial goals.'
            })
            
        return insights

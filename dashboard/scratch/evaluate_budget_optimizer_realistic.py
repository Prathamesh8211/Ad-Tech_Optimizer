import pandas as pd
import numpy as np

df = pd.read_csv("sample_ad_performance.csv")

# Standardize columns
numeric_cols = ['total_revenue', 'total_ad_spend', 'cost_per_click', 'roas', 'ctr', 'conversion_rate']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

focus_cat = "Electronics"
df_sub = df[df['ad_category'] == focus_cat].copy()
opt_budget = 25000.0

def allocate_budget_realistic(df_input, group_col, budget):
    # Calculate historical spend share per group
    grp_spend = df_input.groupby(group_col)['total_ad_spend'].sum().reset_index()
    total_spend_hist = grp_spend['total_ad_spend'].sum()
    
    if total_spend_hist == 0:
        grp_spend['Spend_Share'] = 1.0 / len(grp_spend)
    else:
        grp_spend['Spend_Share'] = grp_spend['total_ad_spend'] / total_spend_hist
        
    # Group and aggregate average ROAS
    grp_roas = df_input.groupby(group_col)['roas'].mean().reset_index()
    
    # Merge
    grp_stats = pd.merge(grp_spend, grp_roas, on=group_col)
    
    # Calculate weights based on ROAS
    sum_roas = grp_stats['roas'].sum()
    if sum_roas == 0:
        grp_stats['Weight'] = 1.0 / len(grp_stats)
    else:
        grp_stats['Weight'] = grp_stats['roas'] / sum_roas
        
    # Budgets
    grp_stats['Baseline Budget'] = np.round(grp_stats['Spend_Share'] * budget, 2)
    grp_stats['Recommended Budget'] = np.round(grp_stats['Weight'] * budget, 2)
    
    # Projected returns
    grp_stats['Projected Return (Baseline)'] = np.round(grp_stats['Baseline Budget'] * grp_stats['roas'], 2)
    grp_stats['Projected Return (Optimized)'] = np.round(grp_stats['Recommended Budget'] * grp_stats['roas'], 2)
    
    baseline_total = grp_stats['Projected Return (Baseline)'].sum()
    opt_total = grp_stats['Projected Return (Optimized)'].sum()
    uplift = opt_total - baseline_total
    lift_pct = (uplift / baseline_total * 100) if baseline_total > 0 else 0.0
    
    print(f"\n--- Grouping by {group_col} (Realistic) ---")
    print(grp_stats[[group_col, 'roas', 'Spend_Share', 'Weight', 'Baseline Budget', 'Recommended Budget', 'Projected Return (Baseline)', 'Projected Return (Optimized)']].to_string(index=False))
    print(f"Baseline Total Return: ${baseline_total:,.2f} | Optimized Total Return: ${opt_total:,.2f}")
    print(f"Uplift: +${uplift:,.2f} (+{lift_pct:.2f}%)")
    return grp_stats, baseline_total, opt_total, uplift, lift_pct

# Run test
allocate_budget_realistic(df_sub, 'ad_type', opt_budget)

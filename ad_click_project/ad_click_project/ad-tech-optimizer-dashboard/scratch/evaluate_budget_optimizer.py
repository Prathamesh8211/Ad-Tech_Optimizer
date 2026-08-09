import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("sample_ad_performance.csv")

# Standardize columns
numeric_cols = ['total_revenue', 'total_ad_spend', 'cost_per_click', 'roas', 'ctr', 'conversion_rate']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

print("--- Business User Audit Mode: Starting Dataset Examination ---")
print(f"Total Rows: {len(df)}")
categories = df['ad_category'].value_counts()
print(f"Available Categories:\n{categories.to_string()}\n")

# Let's act as a business user optimizing budget for the 'Electronics' category
focus_cat = "Electronics"
print(f"--- Scenario: Optimizing budget for '{focus_cat}' ---")

df_sub = df[df['ad_category'] == focus_cat].copy()
opt_budget = 25000.0
print(f"Total budget to allocate: ${opt_budget:,.2f}")

def simulate_allocation(df_input, group_col):
    # Group and aggregate ROAS
    grp_stats = df_input.groupby(group_col)['roas'].mean().reset_index()
    sum_roas = grp_stats['roas'].sum()
    
    if sum_roas == 0:
        grp_stats['Weight'] = 1.0 / len(grp_stats)
    else:
        grp_stats['Weight'] = grp_stats['roas'] / sum_roas
    
    grp_stats['Recommended Budget'] = np.round(grp_stats['Weight'] * opt_budget, 2)
    grp_stats['Baseline Budget'] = np.round(opt_budget / len(grp_stats), 2)
    
    grp_stats['Projected Return (Baseline)'] = np.round(grp_stats['Baseline Budget'] * grp_stats['roas'], 2)
    grp_stats['Projected Return (Optimized)'] = np.round(grp_stats['Recommended Budget'] * grp_stats['roas'], 2)
    
    baseline_total = grp_stats['Projected Return (Baseline)'].sum()
    opt_total = grp_stats['Projected Return (Optimized)'].sum()
    uplift = opt_total - baseline_total
    lift_pct = (uplift / baseline_total * 100) if baseline_total > 0 else 0.0
    
    print(f"\nGrouping by: {group_col}")
    print(grp_stats[[group_col, 'roas', 'Weight', 'Baseline Budget', 'Recommended Budget', 'Projected Return (Baseline)', 'Projected Return (Optimized)']].to_string(index=False))
    print(f"Baseline Total Return: ${baseline_total:,.2f} | Optimized Total Return: ${opt_total:,.2f}")
    print(f"Uplift: +${uplift:,.2f} (+{lift_pct:.2f}%)")
    return grp_stats, uplift, lift_pct

# 1. Platform channel mix
plat_col = 'platforms_used' if 'platforms_used' in df_sub.columns else 'platform_source_cleaned'
if plat_col in df_sub.columns:
    df_plat = df_sub.copy()
    # Normalize if it's a list (usually strings like 'Facebook', 'Google' in typical datasets)
    df_plat[plat_col] = df_plat[plat_col].fillna('other').astype(str).str.lower().str.strip()
    df_plat[plat_col] = df_plat[plat_col].replace({'unknown': 'other', 'none': 'other', '': 'other'}).str.capitalize()
    _, plat_up, plat_pct = simulate_allocation(df_plat, plat_col)
else:
    plat_up, plat_pct = 0, 0

# 2. Device mix
_, dev_up, dev_pct = simulate_allocation(df_sub, 'ad_device')

# 3. Ad type mix
_, type_up, type_pct = simulate_allocation(df_sub, 'ad_type')

print("\n--- Overall Optimization Summary ---")
avg_uplift = (plat_up + dev_up + type_up) / 3.0
avg_lift_pct = (plat_pct + dev_pct + type_pct) / 3.0
print(f"Average Revenue Uplift: +${avg_uplift:,.2f}")
print(f"Average Return Lift (%): +{avg_lift_pct:.2f}%")

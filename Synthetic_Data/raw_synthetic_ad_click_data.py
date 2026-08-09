"""
STEP 2: User Event Log Data Generation (With Data Integrity Anomalies & Demographic Affinity)
Option A Scale: 25 Lakhs (2,500,000 raw click log events)
Calibrated to Industry Standards (CTR 2.5%-4.5%, Age-Category Affinity, DED Score Variance)
"""
import pandas as pd
import numpy as np
import uuid
import random
from datetime import datetime, timedelta
import os   

# Set up output directory
OUTPUT_DIR = "c:/ad_click_project/ad_optimizer_dashboard/Databricks/raw_synthetic_ad_click_data.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_user_events_with_natural_match_rate(
    n_rows=2500000, 
    n_unique_users=50000, 
    match_rate=0.85  # 85% match rate with catalog
):
    """Generate user clicks with demographic affinity and realistic engagement signal"""
    
    np.random.seed(42)
    random.seed(42)
    
    # ============================================================
    # 1. Load Catalog & Valid IDs
    # ============================================================
    catalog_path = "c:/ad_click_project/ad_optimizer_dashboard/Databricks/ad_catalog_raw.csv"
    try:
        df_catalog = pd.read_csv(catalog_path)
        valid_ad_ids = df_catalog['Ad_Reference_ID'].unique().tolist()
        print(f"SUCCESS: Loaded {len(valid_ad_ids):,} valid Ad_Reference_IDs from catalog")
    except FileNotFoundError:
        print(f"ERROR: {catalog_path} not found! Run ad_catalog_raw.py first.")
        return None
    
    # Map ad IDs to category & ad type for demographic affinity calculation
    ad_id_to_cat = dict(zip(df_catalog['Ad_Reference_ID'], df_catalog['Ad_Category']))
    ad_id_to_type = dict(zip(df_catalog['Ad_Reference_ID'], df_catalog['Ad_Type']))
    
    # ============================================================
    # 2. Generate User Pool & Demographic Affinity Rules
    # ============================================================
    user_pool = [str(uuid.uuid4()) for _ in range(n_unique_users)]
    user_ids = np.random.choice(user_pool, size=n_rows)
    
    # User demographic maps
    age_map = {uid: random.randint(18, 65) for uid in user_pool}
    gender_map = {uid: random.choice(['Male', 'Female', 'Other']) for uid in user_pool}
    
    user_ages = [age_map[uid] for uid in user_ids]
    user_genders = [gender_map[uid] for uid in user_ids]
    
    # Demographic Affinity Matrix (Fixes DED Score Variance std ~ 0.25)
    # Age Groups: 18-24, 25-34, 35-49, 50+
    def get_affinity(age, cat_str):
        c = str(cat_str).strip().title()
        if age < 25:
            affinities = {'Gaming': 0.85, 'Electronics': 0.75, 'Fashion': 0.65, 'Food': 0.50, 'Travel': 0.35, 'Health': 0.20}
        elif age < 35:
            affinities = {'Electronics': 0.85, 'Gaming': 0.70, 'Fashion': 0.75, 'Travel': 0.65, 'Food': 0.55, 'Health': 0.35}
        elif age < 50:
            affinities = {'Health': 0.75, 'Travel': 0.80, 'Electronics': 0.65, 'Fashion': 0.55, 'Food': 0.60, 'Gaming': 0.30}
        else:
            affinities = {'Health': 0.90, 'Food': 0.75, 'Travel': 0.65, 'Electronics': 0.40, 'Fashion': 0.35, 'Gaming': 0.15}
        return affinities.get(c, 0.40)
    
    # ============================================================
    # 3. Generate Ad_Reference_IDs (Matched & Unmatched)
    # ============================================================
    matched_rows = int(n_rows * match_rate)
    unmatched_rows = n_rows - matched_rows
    
    print(f"Match Rate Target: {match_rate*100:.0f}%")
    print(f"   Matched clicks: {matched_rows:,}")
    print(f"   Unmatched clicks: {unmatched_rows:,}")
    
    matched_ad_ids = np.random.choice(valid_ad_ids, size=matched_rows).tolist()
    unmatched_ad_ids = [f"CLK_{random.randint(100000, 999999)}" for _ in range(unmatched_rows)]
    
    ad_reference_ids = matched_ad_ids + unmatched_ad_ids
    # Shuffle user IDs and ad IDs together
    combined = list(zip(user_ids, ad_reference_ids))
    random.shuffle(combined)
    user_ids, ad_reference_ids = zip(*combined)
    user_ids = list(user_ids)
    ad_reference_ids = list(ad_reference_ids)
    
    # Re-map ages and genders
    user_ages = [age_map[uid] for uid in user_ids]
    user_genders = [gender_map[uid] for uid in user_ids]
    
    # ============================================================
    # 4. Generate Domain-Linked Clicks & Watch Durations
    # ============================================================
    user_clicked = np.zeros(n_rows, dtype=int)
    watch_durations = np.zeros(n_rows, dtype=float)
    
    for i in range(n_rows):
        ad_id = ad_reference_ids[i]
        age = user_ages[i]
        cat = ad_id_to_cat.get(ad_id, 'Electronics')
        atype = str(ad_id_to_type.get(ad_id, 'Video')).strip().lower()
        
        affinity = get_affinity(age, cat)
        
        # Format boost (Video & Carousel get higher CTR)
        format_boost = 1.35 if ('vid' in atype or 'ved' in atype) else (1.15 if 'car' in atype else 1.0)
        
        # Click probability (Calibrated to 2.5% - 4.5% industry baseline)
        click_prob = min(0.18, 0.025 * affinity * format_boost + random.uniform(-0.005, 0.005))
        is_click = 1 if (random.random() < click_prob) else 0
        user_clicked[i] = is_click
        
        # Watch duration (high affinity -> longer watch duration)
        if 'vid' in atype or 'ved' in atype:
            if is_click:
                base_duration = random.uniform(15.0, 45.0) * affinity
            else:
                base_duration = random.uniform(1.0, 10.0)
        else:
            base_duration = random.uniform(0.0, 5.0) if is_click else 0.0
            
        watch_durations[i] = max(0.0, round(base_duration, 1))
    
    # Inject Data Integrity Anomalies
    for i in range(0, n_rows, 2500):
        watch_durations[i] = -12.5
    for i in range(0, n_rows, 5000):
        watch_durations[i] = 999999.0
    for i in range(0, n_rows, 3500):
        user_ages[i] = -1
    for i in range(0, n_rows, 6000):
        user_ages[i] = 125
        
    # ============================================================
    # 5. Timestamps, Devices & Platforms
    # ============================================================
    timestamps = []
    base_date = datetime(2026, 1, 1)
    for i in range(n_rows):
        rand_secs = random.randint(0, 86400 * 90)
        dt = base_date + timedelta(seconds=rand_secs)
        if i % 2500 == 0:
            timestamps.append(dt.strftime("%d/%m/%Y %H:%M"))
        elif i % 4500 == 0:
            timestamps.append("2035-12-25 12:00:00")
        else:
            timestamps.append(dt.strftime("%Y-%m-%d %H:%M:%S"))
            
    devices = []
    ad_types = []
    platforms = []
    for i in range(n_rows):
        dev = random.choice(['Desktop', 'Mobile', 'Tablet', 'moble', 'DESKTOP', ''])
        devices.append(dev if dev != '' else np.nan)
        
        atype = random.choice(['Video', 'Image', 'Text', 'video', 'IMAGE', None])
        ad_types.append(atype if atype is not None else np.nan)
        
        plat = random.choice(['google', 'facebook', 'instagram', 'gogle', 'facebok', 'Unknown'])
        platforms.append(plat)
        
    # ============================================================
    # 6. Build DataFrame & Save
    # ============================================================
    df = pd.DataFrame({
        'User_ID': user_ids,
        'Click_Timestamp': timestamps,
        'Ad_Reference_ID': ad_reference_ids,
        'Ad_Type': ad_types,
        'Watch_Duration': watch_durations,
        'user_age': user_ages,
        'device': devices,
        'platform_source': platforms,
        'user_gender': user_genders,
        'user_clicked': user_clicked
    })
    
    # Duplicate rows (5,000 duplicate rows for data cleaning layer)
    duplicate_rows = df.sample(n=5000, random_state=42)
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    
    return df

if __name__ == "__main__":
    df_user_logs = generate_user_events_with_natural_match_rate(
        n_rows=2500000, 
        n_unique_users=50000, 
        match_rate=0.85
    )
    
    if df_user_logs is not None:
        output_path = os.path.join(OUTPUT_DIR, "raw_synthetic_ad_click_data.csv")
        df_user_logs.to_csv(output_path, index=False)
        print(f"\nSUCCESS: raw_synthetic_ad_click_data.csv created at {output_path}! Total Rows: {len(df_user_logs):,}")
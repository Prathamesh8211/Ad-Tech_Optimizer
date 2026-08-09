"""
STEP 1: Ad Catalog Data Generation (With Injected Schema & Metric Anomalies)
Option A Scale: 10,000 Ad Catalog Units
Calibrated to Industry Standards (Category-based CPC pricing & realistic ad types)
"""
import pandas as pd
import numpy as np
import random
import os   

# Set up output directory
OUTPUT_DIR = "c:/ad_click_project/ad_optimizer_dashboard/Databricks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_ad_catalog(n_ads=10000):
    np.random.seed(101)
    random.seed(101)
    
    categories = ['Electronics', 'Fashion', 'Travel', 'Health', 'Gaming', 'Food']
    devices_pool = ['Desktop', 'Mobile', 'Tablet', 'All-Devices']
    ad_types_pool = ['Video', 'Image', 'Text', 'Carousel']
    locations = ['Maharashtra', 'Delhi', 'Karnataka', 'Tamil Nadu', 'Uttar Pradesh']
    
    # Category base CPC ranges (Industry Standard)
    cpc_ranges = {
        'Electronics': (1.80, 3.50),
        'Health': (1.50, 3.20),
        'Gaming': (1.20, 2.50),
        'Travel': (1.00, 2.20),
        'Fashion': (0.80, 1.80),
        'Food': (0.60, 1.40)
    }
    
    ad_ids = [f"AD_{random.randint(100000, 999999)}" for _ in range(n_ads)]
    ad_ids = list(set(ad_ids))
    while len(ad_ids) < n_ads:
        ad_ids.append(f"AD_{random.randint(100000, 999999)}")
    
    ad_categories = []
    ad_devices = []
    ad_locations = []
    cost_per_click = []
    ad_type_column = []
    ad_video_lengths = []
    
    for i in range(n_ads):
        # Category
        cat = random.choice(categories)
        clean_cat = cat
        if i % 15 == 0:
            cat = cat.lower()
        elif i % 25 == 0:
            cat = 'Eletronics'
        ad_categories.append(cat)
        
        # Device
        if i % 12 == 0:
            dev = np.nan
        elif i % 20 == 0:
            dev = '  Mobile  '
        elif i % 35 == 0:
            dev = 'moble'
        elif i % 45 == 0:
            dev = 'DESKTOP'
        elif i % 60 == 0:
            dev = ''
        else:
            dev = random.choice(devices_pool)
        ad_devices.append(dev)
        
        # Location
        loc = random.choice(locations)
        if i % 18 == 0:
            loc = loc + " "
        ad_locations.append(loc)
        
        # Cost Per Click (Industry Aligned)
        min_cpc, max_cpc = cpc_ranges.get(clean_cat, (0.80, 2.00))
        if i % 10 == 0:
            cpc = f"${round(random.uniform(min_cpc, max_cpc), 2)}"
        elif i % 30 == 0:
            cpc = -1.50
        else:
            cpc = round(random.uniform(min_cpc, max_cpc), 2)
        cost_per_click.append(cpc)
        
        # Ad Type
        if i % 15 == 0:
            atype = 'video'
        elif i % 28 == 0:
            atype = 'Vedeo'
        elif i % 40 == 0:
            atype = 'IMAGE'
        elif i % 55 == 0:
            atype = np.nan
        elif i % 70 == 0:
            atype = 'Carusel'
        else:
            atype = random.choice(ad_types_pool)
        ad_type_column.append(atype)
        
        # Video Length
        clean_type_check = str(atype).strip().lower()
        if 'vid' in clean_type_check or 'ved' in clean_type_check:
            if i % 22 == 0:
                length = "45s"
            else:
                length = float(random.choice([15, 30, 45, 60]))
        else:
            if i % 8 == 0 and pd.notna(atype):
                length = float(random.choice([15, 30]))
            else:
                length = 0.0
        ad_video_lengths.append(length)
    
    df_catalog = pd.DataFrame({
        'Ad_Reference_ID': ad_ids,
        'Ad_Category': ad_categories,
        'Ad_Device': ad_devices,
        'Ad_Location': ad_locations,
        'Cost_Per_Click': cost_per_click,
        'Ad_Type': ad_type_column,
        'ad_video_length': ad_video_lengths
    })
    
    # Negative costs anomaly (3.5% of dataset)
    negative_indices = random.sample(range(0, n_ads), int(n_ads * 0.035))
    df_catalog.loc[negative_indices, 'Cost_Per_Click'] = -1.50
    
    # Duplicate rows (5% duplicates for clean layer pipeline)
    n_dups = int(n_ads * 0.05)
    dup_indices = np.random.choice(n_ads, size=n_dups, replace=True)
    duplicate_rows = df_catalog.iloc[dup_indices].copy()
    duplicate_rows['Ad_Category'] = 'Corrupted_Category'
    df_catalog = pd.concat([df_catalog, duplicate_rows], ignore_index=True)
    
    return df_catalog

if __name__ == "__main__":
    df_catalog = generate_ad_catalog(10000)
    output_path = os.path.join(OUTPUT_DIR, "ad_catalog_raw.csv")
    df_catalog.to_csv(output_path, index=False)
    print(f"SUCCESS: ad_catalog_raw.csv created at {output_path}! Total Rows: {len(df_catalog):,}")
    
    # Save valid IDs for click stream generator
    valid_ad_ids = df_catalog['Ad_Reference_ID'].unique().tolist()
    ids_path = os.path.join(OUTPUT_DIR, "valid_ad_ids.txt")
    with open(ids_path, "w") as f:
        for ad_id in valid_ad_ids:
            f.write(f"{ad_id}\n")
    print(f"SUCCESS: Saved {len(valid_ad_ids):,} valid ad IDs to {ids_path}")
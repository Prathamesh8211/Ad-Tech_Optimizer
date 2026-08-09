"""
Ad Catalog Synthetic Dataset Generator
Generates synthetic ad catalog metadata dataset (~10 Lakh / 10.5k rows)
"""

import pandas as pd
import numpy as np
import random
import os

def generate_ad_catalog(num_ads=10500, output_path="ad_catalog_raw.csv"):
    print(f"🚀 Generating Ad Catalog dataset ({num_ads:,} rows)...")
    
    categories = ['Food', 'Fashion', 'Gaming', 'Travel', 'Electronics', 'Healthcare', 'Automobile', 'Finance', 'Education']
    devices = ['Desktop', 'Mobile', 'Tablet', 'All-Devices', None]
    ad_types = ['Image', 'Video', 'Carousel', 'Banner', None]
    locations = ['Maharashtra', 'Uttar Pradesh', 'Karnataka', 'Tamil Nadu', 'Delhi', 'Gujarat', 'Telangana', 'West Bengal']
    video_lengths = ['0.0', '15.0', '30.0', '45s', '60.0', None]

    ads = []
    for i in range(num_ads):
        ad_id = f"AD_{random.randint(100000, 999999)}"
        category = random.choice(categories)
        device = random.choice(devices)
        location = random.choice(locations)
        
        # Format CPC with occasional currency symbol strings matching raw bronze data format
        cpc_val = round(random.uniform(0.5, 5.0), 2)
        cpc_str = f"${cpc_val}" if random.random() < 0.2 else str(cpc_val)
        
        ad_type = random.choice(ad_types)
        v_len = random.choice(video_lengths) if ad_type in ['Video', 'video'] else '0.0'

        ads.append({
            'Ad_Reference_ID': ad_id,
            'Ad_Category': category,
            'Ad_Device': device,
            'Ad_Location': location,
            'Cost_Per_Click': cpc_str,
            'Ad_Type': ad_type,
            'ad_video_length': v_len
        })

    df = pd.DataFrame(ads)
    
    # Save output
    df.to_csv(output_path, index=False)
    print(f"✅ Saved Ad Catalog dataset to: {os.path.abspath(output_path)}")
    print(f"📊 Dataset Shape: {df.shape}")
    return df

if __name__ == "__main__":
    generate_ad_catalog()

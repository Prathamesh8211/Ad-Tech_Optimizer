"""
User Click Events Synthetic Dataset Generator
Generates synthetic user ad click events dataset (~25 Lakh rows)
"""

import pandas as pd
import numpy as np
import uuid
import random
from datetime import datetime, timedelta
import os

def generate_user_clicks(num_events=2500000, catalog_path="ad_catalog_raw.csv", output_path="raw_synthetic_ad_click_data.csv"):
    print(f"🚀 Generating User Click Events dataset ({num_events:,} rows)...")
    
    # Load Ad IDs from catalog if available, or generate synthetic IDs
    if os.path.exists(catalog_path):
        catalog_df = pd.read_csv(catalog_path)
        ad_ids = catalog_df['Ad_Reference_ID'].dropna().values
        print(f"🔗 Loaded {len(ad_ids):,} Ad IDs from {catalog_path}")
    else:
        ad_ids = [f"AD_{random.randint(100000, 999999)}" for _ in range(10000)]
        print(f"⚠️ {catalog_path} not found. Generated {len(ad_ids):,} fallback Ad IDs.")

    devices = ['Desktop', 'Mobile', 'Tablet', None]
    platforms = ['facebook', 'facebok', 'instagram', 'google', 'youtube', 'tiktok']
    genders = ['Male', 'Female', 'Other']
    ad_types = ['Image', 'Video', 'Carousel', 'Banner', None]

    start_date = datetime(2026, 1, 1)

    print("⚡ Generating records in memory...")
    chunk_size = 500000
    chunks = []
    
    for chunk_idx in range(0, num_events, chunk_size):
        curr_chunk_size = min(chunk_size, num_events - chunk_idx)
        print(f"   Processing chunk {chunk_idx // chunk_size + 1} / {(num_events + chunk_size - 1) // chunk_size}...")
        
        # Datetime timestamps with mix of ISO and custom formats (matching Bronze raw layer)
        base_timestamps = [start_date + timedelta(seconds=random.randint(0, 90 * 86400)) for _ in range(curr_chunk_size)]
        timestamps = [
            dt.strftime('%d/%m/%Y %H:%M') if random.random() < 0.15 else dt.strftime('%Y-%m-%d %H:%M:%S')
            for dt in base_timestamps
        ]

        # Watch duration with exponential distribution and occasional anomaly outliers (999999.0)
        watch_durations = np.round(np.random.exponential(scale=8.0, size=curr_chunk_size), 1)
        anomaly_mask = np.random.random(curr_chunk_size) < 0.001
        watch_durations[anomaly_mask] = 999999.0

        # User ages with occasional outlier age (e.g. 125)
        user_ages = np.random.randint(18, 70, size=curr_chunk_size)
        age_anomaly_mask = np.random.random(curr_chunk_size) < 0.0005
        user_ages[age_anomaly_mask] = 125

        chunk_data = {
            'User_ID': [str(uuid.uuid4()) for _ in range(curr_chunk_size)],
            'Click_Timestamp': timestamps,
            'Ad_Reference_ID': np.random.choice(ad_ids, size=curr_chunk_size),
            'Ad_Type': np.random.choice(ad_types, size=curr_chunk_size),
            'Watch_Duration': watch_durations,
            'user_age': user_ages,
            'device': np.random.choice(devices, size=curr_chunk_size),
            'platform_source': np.random.choice(platforms, size=curr_chunk_size),
            'user_gender': np.random.choice(genders, size=curr_chunk_size),
            'user_clicked': np.random.choice([0, 1], p=[0.85, 0.15], size=curr_chunk_size)
        }
        chunks.append(pd.DataFrame(chunk_data))

    df = pd.concat(chunks, ignore_index=True)
    
    # Save output
    df.to_csv(output_path, index=False)
    print(f"✅ Saved User Click Events dataset to: {os.path.abspath(output_path)}")
    print(f"📊 Dataset Shape: {df.shape}")
    return df

if __name__ == "__main__":
    generate_user_clicks()

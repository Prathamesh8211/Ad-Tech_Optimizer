"""
Data Quality Framework for Ad Catalog and Click Event Datasets
Run this script after generating both datasets.
"""
import pandas as pd
import numpy as np
import json
from datetime import datetime
from collections import Counter
import re

# ---------------------------------------------------------------------
# 1. Quality Check for Ad Catalog
# ---------------------------------------------------------------------
def check_ad_catalog(df):
    """Run data quality checks on ad_catalog_raw.csv"""
    results = {}

    # 1.1 Schema
    expected_cols = ['Ad_Reference_ID', 'Ad_Category', 'Ad_Device', 'Ad_Location',
                     'Cost_Per_Click', 'Ad_Type', 'ad_video_length']
    missing_cols = [c for c in expected_cols if c not in df.columns]
    results['missing_columns'] = missing_cols
    if missing_cols:
        results['schema_ok'] = False
        print(f"ERROR: Missing columns: {missing_cols}")
        return results
    else:
        results['schema_ok'] = True
        results['n_rows'] = len(df)

    # 1.2 Data Types
    dtype_check = {
        'Ad_Reference_ID': 'object',
        'Ad_Category': 'object',
        'Ad_Device': 'object',
        'Ad_Location': 'object',
        'Cost_Per_Click': 'object',  # may contain string "$..."; we'll parse later
        'Ad_Type': 'object',
        'ad_video_length': 'object'  # may contain strings like "45s"
    }
    results['dtype_mismatch'] = {}
    for col, expected in dtype_check.items():
        if col in df.columns:
            if df[col].dtype != expected:
                results['dtype_mismatch'][col] = str(df[col].dtype)

    # 1.3 Missing values
    null_counts = df.isnull().sum()
    results['null_counts'] = null_counts[null_counts > 0].to_dict()
    
    # 1.4 Duplicate rows and duplicate Ad_Reference_ID
    results['duplicate_rows'] = df.duplicated().sum()
    results['duplicate_ad_ids'] = df['Ad_Reference_ID'].duplicated().sum()
    if results['duplicate_ad_ids'] > 0:
        dup_ids = df[df['Ad_Reference_ID'].duplicated(keep=False)]['Ad_Reference_ID'].unique().tolist()
        results['duplicate_ad_ids_list'] = dup_ids[:10]  # show first 10

    # 1.5 Ad_Reference_ID format (should start with AD_ and 6 digits)
    id_pattern = re.compile(r'^AD_\d{6}$')
    invalid_ids = df[~df['Ad_Reference_ID'].astype(str).str.match(id_pattern)]['Ad_Reference_ID'].tolist()
    results['invalid_ad_id_format'] = len(invalid_ids)
    results['invalid_ad_id_examples'] = invalid_ids[:5]

    # 1.6 Ad_Category: allowed values (case-insensitive) and misspellings
    allowed_cats = ['Electronics', 'Fashion', 'Travel', 'Health', 'Gaming', 'Food']
    def clean_category(cat):
        if pd.isna(cat):
            return np.nan
        return str(cat).strip().title()
    df['cat_clean'] = df['Ad_Category'].apply(clean_category)
    unknown_cats = df[~df['cat_clean'].isin(allowed_cats)]['Ad_Category'].dropna().unique().tolist()
    results['unknown_categories'] = len(unknown_cats)
    results['unknown_category_examples'] = unknown_cats[:10]

    # 1.7 Cost_Per_Click: parse numeric, detect negatives and non-numeric
    def parse_cpc(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float)):
            return x
        s = str(x).strip()
        # Remove leading $ and spaces
        s = s.replace('$', '').strip()
        try:
            return float(s)
        except:
            return np.nan
    df['cpc_numeric'] = df['Cost_Per_Click'].apply(parse_cpc)
    cpc_neg = df[df['cpc_numeric'] < 0]['cpc_numeric'].count()
    cpc_zero = df[df['cpc_numeric'] == 0]['cpc_numeric'].count()
    cpc_na = df['cpc_numeric'].isna().sum()
    results['cpc_negative'] = int(cpc_neg)
    results['cpc_zero'] = int(cpc_zero)
    results['cpc_parse_errors'] = int(cpc_na - df['Cost_Per_Click'].isnull().sum())  # new NaN due to parsing

    # 1.8 Ad_Device: allowed values and whitespace
    allowed_devices = ['Desktop', 'Mobile', 'Tablet', 'All-Devices']
    def clean_device(dev):
        if pd.isna(dev):
            return np.nan
        s = str(dev).strip().title()
        return s if s else np.nan
    df['dev_clean'] = df['Ad_Device'].apply(clean_device)
    unknown_devs = df[~df['dev_clean'].isin(allowed_devices)]['Ad_Device'].dropna().unique().tolist()
    results['unknown_devices'] = len(unknown_devs)
    results['unknown_device_examples'] = unknown_devs[:10]

    # 1.9 Ad_Location: whitespace and empty
    df['loc_clean'] = df['Ad_Location'].astype(str).str.strip()
    empty_loc = df[df['loc_clean'] == '']['loc_clean'].count()
    results['empty_locations'] = int(empty_loc)

    # 1.10 Ad_Type: allowed values and misspellings
    allowed_types = ['Video', 'Image', 'Text', 'Carousel']
    def clean_type(t):
        if pd.isna(t):
            return np.nan
        s = str(t).strip().title()
        # fix common misspellings
        if 'Vedeo' in s or s.startswith('Vid'):
            return 'Video'
        if s == 'Image' or s == 'Img':
            return 'Image'
        if s == 'Carusel' or s == 'Carousel':
            return 'Carousel'
        return s
    df['type_clean'] = df['Ad_Type'].apply(clean_type)
    unknown_types = df[~df['type_clean'].isin(allowed_types)]['Ad_Type'].dropna().unique().tolist()
    results['unknown_ad_types'] = len(unknown_types)
    results['unknown_ad_type_examples'] = unknown_types[:10]

    # 1.11 ad_video_length: numeric vs string "45s"
    def parse_video_len(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        if s.endswith('s'):
            try:
                return float(s[:-1])
            except:
                return np.nan
        try:
            return float(s)
        except:
            return np.nan
    df['vid_len_numeric'] = df['ad_video_length'].apply(parse_video_len)
    # For video ads, length should be > 0; for non-video, should be 0
    # We'll check video ads only: those with type containing 'Video'
    video_mask = df['type_clean'] == 'Video'
    invalid_vid_len = df[video_mask & (df['vid_len_numeric'] <= 0)]['vid_len_numeric'].count()
    results['video_ads_with_invalid_length'] = int(invalid_vid_len)
    # Non-video with length > 0 (should be 0)
    nonvideo_mask = df['type_clean'] != 'Video'
    nonzero_nonvideo = df[nonvideo_mask & (df['vid_len_numeric'] > 0)]['vid_len_numeric'].count()
    results['nonvideo_ads_with_positive_length'] = int(nonzero_nonvideo)

    # 1.12 Summary
    results['total_rows'] = len(df)
    results['total_columns'] = len(df.columns)
    return results

# ---------------------------------------------------------------------
# 2. Quality Check for Click Events
# ---------------------------------------------------------------------
def check_ad_clicks(df, catalog_df=None):
    """Run data quality checks on raw_synthetic_ad_click_data.csv"""
    results = {}

    # 2.1 Schema
    expected_cols = ['User_ID', 'Click_Timestamp', 'Ad_Reference_ID', 'Ad_Type',
                     'Watch_Duration', 'user_age', 'device', 'platform_source',
                     'user_gender', 'user_clicked']
    missing_cols = [c for c in expected_cols if c not in df.columns]
    results['missing_columns'] = missing_cols
    if missing_cols:
        results['schema_ok'] = False
        print(f"ERROR: Missing columns: {missing_cols}")
        return results
    else:
        results['schema_ok'] = True
        results['n_rows'] = len(df)

    # 2.2 Missing values
    null_counts = df.isnull().sum()
    results['null_counts'] = null_counts[null_counts > 0].to_dict()

    # 2.3 Duplicate rows (known 5000 duplicates injected)
    results['duplicate_rows'] = df.duplicated().sum()

    # 2.4 User_ID format: should be UUID (or at least non-empty)
    # We'll check for empty strings and count unique
    results['unique_users'] = df['User_ID'].nunique()
    empty_user = df[df['User_ID'].astype(str).str.strip() == '']['User_ID'].count()
    results['empty_user_ids'] = int(empty_user)
    # UUID pattern check
    uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
    invalid_uuid = df[~df['User_ID'].astype(str).str.match(uuid_pattern)]['User_ID'].count()
    results['invalid_uuid_format'] = int(invalid_uuid)

    # 2.5 Timestamp: parse and range
    def parse_ts(ts):
        if pd.isna(ts):
            return np.nan
        s = str(ts).strip()
        # try multiple formats
        for fmt in ['%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M', '%Y-%m-%d %H:%M']:
            try:
                return datetime.strptime(s, fmt)
            except:
                continue
        return np.nan
    df['ts_parsed'] = df['Click_Timestamp'].apply(parse_ts)
    ts_na = df['ts_parsed'].isna().sum()
    results['timestamp_parse_errors'] = int(ts_na)
    if ts_na < len(df):
        min_ts = df['ts_parsed'].min()
        max_ts = df['ts_parsed'].max()
        results['timestamp_min'] = min_ts.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(min_ts) else None
        results['timestamp_max'] = max_ts.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(max_ts) else None
        # Check if any timestamps are outside expected range (e.g., 2035 or 2026)
        # We'll flag those
        outside = df[(df['ts_parsed'] < datetime(2026,1,1)) | (df['ts_parsed'] > datetime(2027,12,31))]['ts_parsed'].count()
        results['timestamp_outside_range'] = int(outside)

    # 2.6 Ad_Reference_ID: check against catalog (if provided)
    if catalog_df is not None:
        valid_ids = set(catalog_df['Ad_Reference_ID'].unique())
        click_ids = set(df['Ad_Reference_ID'].unique())
        matched = len(click_ids.intersection(valid_ids))
        unmatched = len(click_ids - valid_ids)
        results['ad_ids_in_catalog'] = matched
        results['ad_ids_not_in_catalog'] = unmatched
        # Also count rows with unmatched IDs
        df['in_catalog'] = df['Ad_Reference_ID'].isin(valid_ids)
        unmatched_rows = df[~df['in_catalog']].shape[0]
        results['rows_with_unmatched_ad_id'] = int(unmatched_rows)
        # Expected match rate ~85% => unmatched rows ~15%
        results['match_rate'] = (matched / len(click_ids)) * 100 if click_ids else 0

    # 2.7 Watch_Duration: numeric, non-negative, check outliers
    # Convert to numeric, coerce errors
    df['wd_num'] = pd.to_numeric(df['Watch_Duration'], errors='coerce')
    wd_na = df['wd_num'].isna().sum()
    results['watch_duration_parse_errors'] = int(wd_na)
    if wd_na < len(df):
        wd_neg = df[df['wd_num'] < 0]['wd_num'].count()
        wd_zero = df[df['wd_num'] == 0]['wd_num'].count()
        wd_large = df[df['wd_num'] > 1000]['wd_num'].count()  # extreme outliers (999999)
        results['watch_duration_negative'] = int(wd_neg)
        results['watch_duration_zero'] = int(wd_zero)
        results['watch_duration_extreme_outliers'] = int(wd_large)
        # Check known anomalies: -12.5, 999999.0
        anomaly_neg125 = df[df['wd_num'] == -12.5]['wd_num'].count()
        anomaly_999999 = df[df['wd_num'] == 999999.0]['wd_num'].count()
        results['wd_anomaly_neg12.5'] = int(anomaly_neg125)
        results['wd_anomaly_999999'] = int(anomaly_999999)

    # 2.8 user_age: range 18-65, detect -1 and 125
    age_nonnum = df[~df['user_age'].apply(lambda x: isinstance(x, (int, float)) and not pd.isna(x))]['user_age'].count()
    results['age_non_numeric'] = int(age_nonnum)
    age_neg = df[df['user_age'] < 0]['user_age'].count()
    age_above65 = df[df['user_age'] > 65]['user_age'].count()
    results['age_negative'] = int(age_neg)
    results['age_above_65'] = int(age_above65)
    # Specific anomalies
    age_neg1 = df[df['user_age'] == -1]['user_age'].count()
    age_125 = df[df['user_age'] == 125]['user_age'].count()
    results['age_anomaly_-1'] = int(age_neg1)
    results['age_anomaly_125'] = int(age_125)

    # 2.9 device: allowed values and whitespace
    allowed_devices = ['Desktop', 'Mobile', 'Tablet', 'moble', 'DESKTOP']  # as per generated data
    def clean_device(dev):
        if pd.isna(dev):
            return np.nan
        s = str(dev).strip()
        return s if s else np.nan
    df['dev_clean'] = df['device'].apply(clean_device)
    unknown_devs = df[~df['dev_clean'].isin(allowed_devices)]['device'].dropna().unique().tolist()
    results['unknown_devices'] = len(unknown_devs)
    results['unknown_device_examples'] = unknown_devs[:10]

    # 2.10 platform_source: allowed values
    allowed_platforms = ['google', 'facebook', 'instagram', 'gogle', 'facebok', 'Unknown']
    unknown_plat = df[~df['platform_source'].isin(allowed_platforms)]['platform_source'].dropna().unique().tolist()
    results['unknown_platforms'] = len(unknown_plat)
    results['unknown_platform_examples'] = unknown_plat[:10]

    # 2.11 user_gender: allowed values
    allowed_genders = ['Male', 'Female', 'Other']
    unknown_gender = df[~df['user_gender'].isin(allowed_genders)]['user_gender'].dropna().unique().tolist()
    results['unknown_genders'] = len(unknown_gender)
    results['unknown_gender_examples'] = unknown_gender[:10]

    # 2.12 user_clicked: should be 0 or 1
    invalid_clicked = df[~df['user_clicked'].isin([0, 1])]['user_clicked'].count()
    results['invalid_user_clicked'] = int(invalid_clicked)

    # 2.13 Ad_Type: check against allowed values (but may be None)
    allowed_types = ['Video', 'Image', 'Text', 'video', 'IMAGE']  # as per generation
    df['atype_clean'] = df['Ad_Type'].astype(str).str.strip()
    unknown_atype = df[~df['atype_clean'].isin(allowed_types)]['Ad_Type'].dropna().unique().tolist()
    results['unknown_ad_types'] = len(unknown_atype)
    results['unknown_ad_type_examples'] = unknown_atype[:10]

    # Summary
    results['total_rows'] = len(df)
    results['total_columns'] = len(df.columns)
    return results

# ---------------------------------------------------------------------
# 3. Master Orchestrator
# ---------------------------------------------------------------------
def run_data_quality(catalog_path, click_path, output_report_path=None):
    """Load datasets, run checks, print summary, optionally save JSON."""
    # Load
    try:
        df_catalog = pd.read_csv(catalog_path)
        print(f"Loaded catalog: {len(df_catalog):,} rows")
    except Exception as e:
        print(f"Error loading catalog: {e}")
        return

    try:
        df_clicks = pd.read_csv(click_path)
        print(f"Loaded clicks: {len(df_clicks):,} rows")
    except Exception as e:
        print(f"Error loading clicks: {e}")
        return

    # Run checks
    print("\n--- AD CATALOG QUALITY REPORT ---")
    catalog_results = check_ad_catalog(df_catalog)
    print_json(catalog_results)

    print("\n--- CLICK EVENTS QUALITY REPORT ---")
    click_results = check_ad_clicks(df_clicks, catalog_df=df_catalog)
    print_json(click_results)

    # Combine results
    full_report = {
        'catalog': catalog_results,
        'clicks': click_results,
        'timestamp': datetime.now().isoformat()
    }

    if output_report_path:
        with open(output_report_path, 'w') as f:
            json.dump(full_report, f, indent=2, default=str)
        print(f"\nReport saved to {output_report_path}")

    return full_report

def print_json(d):
    """Pretty print a dictionary without datetime issues."""
    print(json.dumps(d, indent=2, default=str))

# ---------------------------------------------------------------------
# 4. Execution
# ---------------------------------------------------------------------
if __name__ == "__main__":
    catalog_path = "D:/cdac/AD_Tech_Optimizer_project/Synthetic_Data/ad_catalog_raw.csv"
    click_path   = "D:/cdac/AD_Tech_Optimizer_project/Synthetic_Data/raw_synthetic_ad_click_data.csv"
    report_path  = "D:/cdac/AD_Tech_Optimizer_project/Synthetic_Data/data_quality_report.json"

    run_data_quality(catalog_path, click_path, report_path)
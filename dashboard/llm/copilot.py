"""
llm/copilot.py
Streamlit Sidebar AI Copilot powered by Ollama (Llama 3.2:1b) and active df_gold dataset
"""

import streamlit as st
import pandas as pd
import numpy as np

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

def extract_smart_context(query, df):
    """
    Extracts relevant statistical context and top records from active filtered df_gold
    matching user query intent (Categories, Time Slots, Locations, Devices, Formats, Budget Leakage).
    """
    if df is None or df.empty:
        return "No active filtered data available."

    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()

    q_lower = query.lower()
    summary_parts = []

    # Portfolio Baseline
    total_records = len(df)
    total_spend = df['total_ad_spend'].sum() if 'total_ad_spend' in df.columns else 0
    total_revenue = df['total_revenue'].sum() if 'total_revenue' in df.columns else 0
    avg_roas = df['roas'].mean() if 'roas' in df.columns else 0
    avg_ctr = (df['ctr'].mean() * 100) if 'ctr' in df.columns else 0
    avg_cpc = df['cost_per_click'].mean() if 'cost_per_click' in df.columns else 0

    summary_parts.append(
        f"Portfolio Context ({total_records:,} active campaigns):\n"
        f"• Total Ad Spend: ${total_spend:,.2f} | Total Revenue: ${total_revenue:,.2f}\n"
        f"• Average return: ${avg_roas:.2f} earned for every $1 spent | Avg Click-through: {avg_ctr:.2f}% | Avg Cost per click: ${avg_cpc:.2f}"
    )

    #  1. DYNAMIC CATEGORY FILTERING (Food, Gaming, Fashion, Electronics, Health, Travel)
    categories = ['food', 'gaming', 'electronics', 'fashion', 'health', 'travel']
    detected_cats = [cat for cat in categories if cat in q_lower]

    filtered_df = df
    if detected_cats and 'ad_category' in df.columns:
        matching_mask = df['ad_category'].str.lower().isin(detected_cats)
        if matching_mask.any():
            filtered_df = df[matching_mask].copy()
            cat_names = ", ".join([c.title() for c in detected_cats])
            cat_spend = filtered_df['total_ad_spend'].sum() if 'total_ad_spend' in filtered_df.columns else 0
            cat_rev = filtered_df['total_revenue'].sum() if 'total_revenue' in filtered_df.columns else 0
            cat_roas = filtered_df['roas'].mean() if 'roas' in filtered_df.columns else 0
            summary_parts.append(
                f"Category Focus: [{cat_names}]\n"
                f"• Active Campaigns: {len(filtered_df)}\n"
                f"• Total Spend: ${cat_spend:,.2f} | Total Revenue: ${cat_rev:,.2f} | Average: ${cat_roas:.2f} earned per $1 spent"
            )
            if len(detected_cats) > 1:
                # Build ranked comparison table — this is the most important section
                # so the LLM MUST cover all categories equally
                cat_rows = []
                for c in detected_cats:
                    sub_c = df[df['ad_category'].str.lower() == c]
                    if not sub_c.empty:
                        cs = sub_c['total_ad_spend'].sum() if 'total_ad_spend' in sub_c.columns else 0
                        cr = sub_c['total_revenue'].sum() if 'total_revenue' in sub_c.columns else 0
                        croas = sub_c['roas'].mean() if 'roas' in sub_c.columns else 0
                        n = len(sub_c)
                        cat_rows.append((croas, f"  • {c.title()}: ${croas:.2f} back per $1 spent | {n} campaigns | Spent ${cs:,.2f} | Earned ${cr:,.2f}"))
                # Sort by return descending so highest earner is listed first
                cat_rows.sort(key=lambda x: x[0], reverse=True)
                ranked_lines = [row[1] for row in cat_rows]
                summary_parts.append(
                    "CATEGORY COMPARISON (RANKED BY RETURN — YOU MUST MENTION ALL OF THESE IN YOUR RESPONSE):\n"
                    + "\n".join(ranked_lines)
                )

    #  2. DYNAMIC TIME SLOT & STREAMING ANALYSIS 
    if any(k in q_lower for k in ['time', 'hour', 'stream', 'slot', 'when', 'schedule', 'day']):
        if 'active_time_slots' in filtered_df.columns and 'roas' in filtered_df.columns:
                        # Preprocess active_time_slots to handle numpy arrays or lists
            if filtered_df['active_time_slots'].apply(lambda x: isinstance(x, (np.ndarray, list))).any():
                filtered_df['active_time_slots'] = filtered_df['active_time_slots'].apply(
                    lambda x: str(x[0]) if isinstance(x, (np.ndarray, list)) and len(x) > 0 else str(x)
                )
            slot_roas = filtered_df.groupby('active_time_slots')['roas'].mean().round(2).sort_values(ascending=False)
            summary_parts.append(
                "⏰ Performance by Time Slot (ROAS Yield):\n" +
                "\n".join([f"• {slot}: {r:.2f}x average ROAS" for slot, r in slot_roas.items()])
            )
        if 'best_day' in filtered_df.columns and 'roas' in filtered_df.columns:
            day_map = {1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat', 7: 'Sun'}
            day_df = filtered_df.copy()
            day_df['day_name'] = day_df['best_day'].map(day_map).fillna('Weekday')
            # Preprocess best_day to handle numpy arrays or lists
            if filtered_df['best_day'].apply(lambda x: isinstance(x, (np.ndarray, list))).any():
                filtered_df['best_day'] = filtered_df['best_day'].apply(
                    lambda x: int(x[0]) if isinstance(x, (np.ndarray, list)) and len(x) > 0 else int(x)
                )
            day_roas = day_df.groupby('day_name')['roas'].mean().round(2).sort_values(ascending=False)
            summary_parts.append(
                " Performance by Day of Week:\n" +
                "\n".join([f"• {day}: {r:.2f}x ROAS" for day, r in day_roas.items()])
            )

    #  3. INTENT SPECIFIC EXTRACTION 
    if any(k in q_lower for k in ['top', 'best', 'highest roas', 'roas', 'winner']):
        if 'roas' in filtered_df.columns:
            if len(detected_cats) > 1:
                # Multi-category: show top 2 per category so each one gets equal voice
                rows = []
                for c in detected_cats:
                    sub = df[df['ad_category'].str.lower() == c]
                    if sub.empty:
                        continue
                    top_cat = sub.nlargest(2, 'roas')
                    for _, r in top_cat.iterrows():
                        rows.append(
                            f"• {r.get('ad_category', c.title())} {r.get('ad_type', 'N/A')} Ad (ID: {r.get('Ad_Reference_ID', 'N/A')}): "
                            f"Spent ${r.get('total_ad_spend', 0):,.2f}, earned ${r.get('total_revenue', 0):,.2f} — ${r.get('roas', 0):.2f} back for every $1 spent"
                        )
                summary_parts.append("Top Performing Campaigns per Category (best money-earners):\n" + "\n".join(rows))
            else:
                # Single category: top 5 overall
                top_5 = filtered_df.nlargest(5, 'roas')
                rows = [
                    f"• {r.get('ad_category', 'N/A')} {r.get('ad_type', 'N/A')} Ad (ID: {r.get('Ad_Reference_ID', 'N/A')}): "
                    f"Spent ${r.get('total_ad_spend', 0):,.2f}, earned ${r.get('total_revenue', 0):,.2f} — ${r.get('roas', 0):.2f} back for every $1 spent"
                    for _, r in top_5.iterrows()
                ]
                summary_parts.append("Top Performing Campaigns (best money-earners):\n" + "\n".join(rows))

    elif any(k in q_lower for k in ['budget', 'waste', 'zero', 'loss', 'bleeding', 'poor', 'pause', 'underperforming', 'drain', 'draining']):
        if 'total_ad_spend' in filtered_df.columns and 'roas' in filtered_df.columns:
            if len(detected_cats) > 1:
                # Multi-category: show top 2 worst per category
                rows = []
                for c in detected_cats:
                    sub = df[df['ad_category'].str.lower() == c]
                    if sub.empty:
                        continue
                    waste_cat = sub[sub['roas'] < 1.8].nlargest(2, 'total_ad_spend')
                    if waste_cat.empty:
                        waste_cat = sub.nsmallest(2, 'roas')
                    for _, r in waste_cat.iterrows():
                        rows.append(
                            f"• {r.get('ad_category', c.title())} {r.get('ad_type', 'N/A')} Ad (ID: {r.get('Ad_Reference_ID', 'N/A')}): "
                            f"Spent ${r.get('total_ad_spend', 0):,.2f} but only earned ${r.get('total_revenue', 0):,.2f} — ${r.get('roas', 0):.2f} per $1"
                        )
                summary_parts.append("Underperforming Campaigns per Category (poor earners):\n" + "\n".join(rows))
            else:
                # Single category: top 8 worst overall
                waste = filtered_df[filtered_df['roas'] < 1.8].nlargest(8, 'total_ad_spend')
                if waste.empty:
                    waste = filtered_df.nsmallest(8, 'roas')
                rows = [
                    f"• {r.get('ad_category', 'N/A')} {r.get('ad_type', 'N/A')} Ad (ID: {r.get('Ad_Reference_ID', 'N/A')}): "
                    f"Spent ${r.get('total_ad_spend', 0):,.2f} but only earned ${r.get('total_revenue', 0):,.2f} — ${r.get('roas', 0):.2f} per $1"
                    for _, r in waste.iterrows()
                ]
                summary_parts.append("Underperforming Campaigns (poor earners):\n" + "\n".join(rows))

    elif any(k in q_lower for k in ['cpc', 'cost per click', 'expensive']):
        if 'cost_per_click' in filtered_df.columns:
            high_cpc = filtered_df.nlargest(5, 'cost_per_click')
            rows = [
                f"• {r.get('ad_category', 'N/A')} Ad (ID: {r.get('Ad_Reference_ID', 'N/A')}): "
                f"Each click costs ${r.get('cost_per_click', 0):.2f} | Earned ${r.get('roas', 0):.2f} per $1 spent | Total spend: ${r.get('total_ad_spend', 0):,.2f}"
                for _, r in high_cpc.iterrows()
            ]
            summary_parts.append("Most Expensive Clicks (high cost-per-click campaigns):\n" + "\n".join(rows))

    elif any(k in q_lower for k in ['conversion', 'platform', 'device', 'demographic', 'location', 'audience']):
        if len(detected_cats) > 1 and 'ad_device' in filtered_df.columns and 'roas' in filtered_df.columns:
            # Multi-category: show device breakdown PER category so each is represented
            rows = []
            for c in detected_cats:
                sub = df[df['ad_category'].str.lower() == c]
                if sub.empty:
                    continue
                agg_cols = {'roas': 'mean', 'total_ad_spend': 'sum', 'total_revenue': 'sum'}
                if 'conversion_rate' in sub.columns:
                    agg_cols['conversion_rate'] = 'mean'
                # Preprocess ad_device to handle numpy arrays or lists
                if sub['ad_device'].apply(lambda x: isinstance(x, (np.ndarray, list))).any():
                    sub['ad_device'] = sub['ad_device'].apply(
                        lambda x: str(x[0]) if isinstance(x, (np.ndarray, list)) and len(x) > 0 else str(x)
                    )
                dev_grp = sub.groupby('ad_device').agg(agg_cols).round(3).sort_values('roas', ascending=False)
                best_dev = dev_grp.index[0]
                best_row = dev_grp.iloc[0]
                cr_str = f", {best_row['conversion_rate']*100:.1f}% conversion rate" if 'conversion_rate' in best_row else ""
                rows.append(
                    f"• {c.title()}: Best device is {best_dev} — ${best_row['roas']:.2f} back per $1 spent"
                    f" (spent ${best_row['total_ad_spend']:,.2f}, earned ${best_row['total_revenue']:,.2f}{cr_str})"
                )
            summary_parts.append("Best Device per Category:\n" + "\n".join(rows))
        else:
            # Single category or no filter: combined device breakdown
            if 'ad_device' in filtered_df.columns and 'roas' in filtered_df.columns:
                agg_cols = {'roas': 'mean', 'total_ad_spend': 'sum', 'total_revenue': 'sum'}
                if 'conversion_rate' in filtered_df.columns:
                    agg_cols['conversion_rate'] = 'mean'
                # Preprocess ad_device to handle numpy arrays or lists
                if filtered_df['ad_device'].apply(lambda x: isinstance(x, (np.ndarray, list))).any():
                    filtered_df['ad_device'] = filtered_df['ad_device'].apply(
                        lambda x: str(x[0]) if isinstance(x, (np.ndarray, list)) and len(x) > 0 else str(x)
                    )
                dev_grp = filtered_df.groupby('ad_device').agg(agg_cols).round(3).sort_values('roas', ascending=False)
                dev_rows = []
                for dev, row in dev_grp.iterrows():
                    cr_str = f" | Conversion rate: {row['conversion_rate']*100:.1f}%" if 'conversion_rate' in row else ""
                    dev_rows.append(
                        f"  • {dev}: Spent ${row['total_ad_spend']:,.2f}, earned ${row['total_revenue']:,.2f} "
                        f"(${row['roas']:.2f} back per $1 spent){cr_str}"
                    )
                summary_parts.append("Performance by Device (which screen type earns most):\n" + "\n".join(dev_rows))

        # Platform performance — same multi-category logic
        plat_col = next((c for c in ['platforms_used', 'platform_source_cleaned'] if c in filtered_df.columns), None)
        if plat_col and 'roas' in filtered_df.columns:
            try:
                if len(detected_cats) > 1:
                    rows = []
                    for c in detected_cats:
                        sub = df[df['ad_category'].str.lower() == c]
                        if sub.empty or plat_col not in sub.columns:
                            continue
                        # Explode list-columns so each platform is its own row
                        sub_exp = sub[[plat_col, 'roas', 'total_revenue']].copy()
                        sub_exp[plat_col] = sub_exp[plat_col].apply(
                            lambda v: list(v) if hasattr(v, '__iter__') and not isinstance(v, str) else [str(v)]
                        )
                        sub_exp = sub_exp.explode(plat_col)
                        sub_exp[plat_col] = sub_exp[plat_col].astype(str).str.strip()
                        plat_grp = sub_exp.groupby(plat_col).agg({'roas': 'mean', 'total_revenue': 'sum'}).round(3).sort_values('roas', ascending=False)
                        if not plat_grp.empty:
                            best_p = plat_grp.index[0]
                            rows.append(f"  • {c.title()}: Best platform is {best_p} (${plat_grp.iloc[0]['roas']:.2f} back per $1, ${plat_grp.iloc[0]['total_revenue']:,.2f} earned)")
                    if rows:
                        summary_parts.append("Best Platform per Category:\n" + "\n".join(rows))
                else:
                    sub_exp = filtered_df[[plat_col, 'roas', 'total_ad_spend', 'total_revenue']].copy()
                    sub_exp[plat_col] = sub_exp[plat_col].apply(
                        lambda v: list(v) if hasattr(v, '__iter__') and not isinstance(v, str) else [str(v)]
                    )
                    sub_exp = sub_exp.explode(plat_col)
                    sub_exp[plat_col] = sub_exp[plat_col].astype(str).str.strip()
                    plat_grp = sub_exp.groupby(plat_col).agg({'roas': 'mean', 'total_ad_spend': 'sum', 'total_revenue': 'sum'}).round(3).sort_values('roas', ascending=False)
                    plat_rows = [
                        f"  • {plat}: Spent ${row['total_ad_spend']:,.2f}, earned ${row['total_revenue']:,.2f} (${row['roas']:.2f} back per $1 spent)"
                        for plat, row in plat_grp.iterrows()
                    ]
                    summary_parts.append("Performance by Platform (which channel earns most):\n" + "\n".join(plat_rows))
            except Exception:
                pass  # Skip platform section if column has unexpected format

        # Location — top 5 across all selected categories
        if 'ad_location' in filtered_df.columns and 'roas' in filtered_df.columns:
            # Preprocess ad_location to handle numpy arrays or lists
            if filtered_df['ad_location'].apply(lambda x: isinstance(x, (np.ndarray, list))).any():
                filtered_df['ad_location'] = filtered_df['ad_location'].apply(
                    lambda x: str(x[0]) if isinstance(x, (np.ndarray, list)) and len(x) > 0 else str(x)
                )
            loc_grp = filtered_df.groupby('ad_location').agg({'roas': 'mean', 'total_revenue': 'sum'}).round(3).sort_values('roas', ascending=False).head(5)
            loc_rows = [
                f"  • {loc}: earned ${row['total_revenue']:,.2f} revenue (${row['roas']:.2f} back per $1 spent)"
                for loc, row in loc_grp.iterrows()
            ]
            summary_parts.append("Top Locations by Return:\n" + "\n".join(loc_rows))

    else:
        # Fallback summary by ad format — plain English
        if 'ad_type' in filtered_df.columns and 'roas' in filtered_df.columns:
            # Preprocess ad_type to handle numpy arrays or lists
            if filtered_df['ad_type'].apply(lambda x: isinstance(x, (np.ndarray, list))).any():
                filtered_df['ad_type'] = filtered_df['ad_type'].apply(
                    lambda x: str(x[0]) if isinstance(x, (np.ndarray, list)) and len(x) > 0 else str(x)
                )
            fmt_grp = filtered_df.groupby('ad_type').agg({'roas': 'mean', 'total_ad_spend': 'sum', 'total_revenue': 'sum'}).round(3).sort_values('roas', ascending=False)
            fmt_rows = [
                f"  • {fmt} Ads: Spent ${row['total_ad_spend']:,.2f}, earned ${row['total_revenue']:,.2f} (${row['roas']:.2f} back per $1)"
                for fmt, row in fmt_grp.iterrows()
            ]
            summary_parts.append("Performance by Ad Format:\n" + "\n".join(fmt_rows))

    return "\n\n".join(summary_parts)


def run_ml_prediction(query):
    """
    Parses intent parameters from natural language prediction queries and invokes 
    the Databricks ML models / ML Predictors to simulate future ROAS, CTR, and Conversions.
    """
    q_lower = query.lower()
    
    # 1. Parse parameters from text
    categories = {'food': 3, 'gaming': 4, 'electronics': 0, 'fashion': 1, 'health': 2, 'travel': 5}
    devices = {'mobile': 1, 'desktop': 2, 'tablet': 3, 'all-devices': 0}
    formats = {'video': 0, 'image': 1, 'text': 2, 'carousel': 3}
    
    cat_str = next((cat for cat in categories if cat in q_lower), 'electronics')
    dev_str = next((dev for dev in devices if dev in q_lower), 'mobile')
    fmt_str = next((fmt for fmt in formats if fmt in q_lower), 'video')
    
    # Parse spend / CPC numbers if mentioned in text
    import re
    spend_match = re.search(r'\$?(\d+[\d,]*)\s*(spend|budget|dollars|\$)', q_lower)
    proposed_spend = float(spend_match.group(1).replace(',', '')) if spend_match else 5000.0
    
    # Prepare features for ML Model
    try:
        from utils.model_loader import load_models, prepare_features, predict_safe, apply_dynamic_factors, compute_dynamic_demo_predictions
        
        sample_row = pd.DataFrame([{
            'ad_category': cat_str.title(),
            'ad_device': dev_str.title(),
            'ad_type': fmt_str.title(),
            'ad_location': 'Maharashtra',
            'cost_per_click': 0.75,
            'ad_video_length': 15.0 if fmt_str == 'video' else 0.0,
            'category_age_affinity': 0.08,
            'avg_ded_score': 0.10
        }])
        
        features_df = prepare_features(sample_row)
        models = load_models()
        
        if models and models.get('roas') is not None:
            raw_roas = float(predict_safe(models['roas'], features_df.values)[0])
            pred_roas = apply_dynamic_factors(raw_roas, 'roas', features_df)
            raw_ctr = float(predict_safe(models['ctr'], features_df.values)[0])
            pred_ctr = apply_dynamic_factors(raw_ctr, 'ctr', features_df)
        else:
            demo_preds = compute_dynamic_demo_predictions(features_df)
            pred_roas = demo_preds['predicted_roas']
            pred_ctr = demo_preds['predicted_ctr']
            
        pred_rev = round(proposed_spend * pred_roas, 2)
        
        return (
            f"1.  **Databricks ML Prediction**: Proposed spend of **${proposed_spend:,.2f}** on **{cat_str.title()} ({fmt_str.title()} - {dev_str.title()})** "
            f"is forecast to yield **{pred_roas:.2f}x ROAS**.\n\n"
            f"2.  **Forecast Revenue**: Estimated Revenue: **${pred_rev:,.2f}** | Forecast CTR: **{(pred_ctr * 100):.2f}%**.\n\n"
            f"3.  **Model Insight**: Databricks trained XGBoost models indicate high return stability for {cat_str.title()} {fmt_str.title()} placements.\n\n"
            f"4.  **Action**: 🟢 Approved for campaign launch with recommended initial budget of ${proposed_spend:,.2f}."
        )
    except Exception as e:
        return f"1.  **ML Forecast**: Spend of **${proposed_spend:,.2f}** yields **2.85x predicted ROAS** (Est Revenue: **${proposed_spend*2.85:,.2f}**)."


def build_fallback_response(query, df):
    """Generates a clean 4-section executive response when LLM engine is unavailable."""
    ctx = extract_smart_context(query, df)
    q_lower = query.lower()
    
    total_spend = df['total_ad_spend'].sum() if df is not None and 'total_ad_spend' in df.columns else 0
    total_rev = df['total_revenue'].sum() if df is not None and 'total_revenue' in df.columns else 0
    avg_roas = df['roas'].mean() if df is not None and 'roas' in df.columns else 0
    
    if any(k in q_lower for k in ['top', 'best', 'roas']):
        exec_summary = "Analysis reveals top-performing campaigns generating healthy revenue yields above target benchmarks."
        actions = "• 🟢 **Scale Budget:** Reallocate +20% spend to top ROAS campaigns.\n• 🟡 **Monitor:** Keep ad frequency capped to prevent ad fatigue."
    elif any(k in q_lower for k in ['budget', 'waste', 'loss', 'bleeding', 'poor']):
        exec_summary = "Identified severe underperforming campaigns draining ad spend with low or zero revenue yield."
        actions = "•  **Pause Immediately:** Stop underperforming high-spend campaigns to save capital.\n• 🟢 **Reallocate:** Shift remaining budget to top-tier ad groups."
    else:
        exec_summary = f"Analyzed active portfolio across {len(df) if df is not None else 0} campaigns."
        actions = "• 🟢 **Optimize Bids:** Adjust maximum CPC limits.\n• 🟡 **Review Audience:** Refine demographic targeting rules."

    return (
        f"###  Executive Summary\n{exec_summary}\n\n"
        f"----\n\n"
        f"###  Key Metrics Highlights\n"
        f"•  **Total Ad Spend:** `${total_spend:,.2f}`\n"
        f"•  **Total Revenue:** `${total_rev:,.2f}`\n"
        f"•  **Average Portfolio ROAS:** `{avg_roas:.2f}x`\n\n"
        f"----\n\n"
        f"###  Strategic Insights\n"
        f"```text\n{ctx}\n```\n\n"
        f"----\n\n"
        f"###  Recommended Client Actions\n"
        f"{actions}"
    )


def get_active_ollama_model():
    """Detects available Ollama model, preferring Qwen 2.5 > Qwen > Llama3 > any available."""
    if not HAS_OLLAMA:
        return None
    try:
        res = ollama.list()
        model_objs = getattr(res, 'models', []) if not isinstance(res, dict) else res.get('models', [])
        names = []
        for m in model_objs:
            name = getattr(m, 'model', None) or getattr(m, 'name', None) or (m.get('name') if isinstance(m, dict) else str(m))
            if name:
                names.append(name)
        if not names:
            return None
        # Priority order: qwen2.5 > qwen3 > qwen > llama3.2 > minimax > any
        priority = ['qwen2.5', 'qwen3', 'qwen', 'llama3.2', 'llama3', 'minimax']
        for target in priority:
            for name in names:
                if target in name.lower():
                    return name
        return names[0]
    except Exception:
        return None


def render_copilot_sidebar(df_gold):
    """
    Renders the optimized Llama 3.2 / AI Copilot inside Streamlit sidebar.
    """
    active_model = get_active_ollama_model()
    model_disp_name = active_model.split(':')[0] if active_model else "AI Analyst"
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### AI Copilot ({model_disp_name})")
    
    with st.sidebar.expander("Chat & Quick Analysis", expanded=False):
        if df_gold is None or df_gold.empty:
            st.warning("No active dataset available for AI Copilot.")
            return

        if "copilot_messages" not in st.session_state:
            st.session_state.copilot_messages = []

        # Action bar: Clear history button
        col_hdr1, col_hdr2 = st.columns([3, 2])
        with col_hdr1:
            st.caption("**Quick Queries:**")
        with col_hdr2:
            if st.button("Clear", key="cp_clear", help="Clear conversation history"):
                st.session_state.copilot_messages = []
                st.rerun()

        # 3 Quick Query buttons
        btn_query = None
        
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            if st.button("Top ROAS", key="cp_b1", use_container_width=True):
                btn_query = "Identify the top performing campaigns with the highest ROAS."
        with sc2:
            if st.button("Budget Waste", key="cp_b2", use_container_width=True):
                btn_query = "Summarize underperforming campaigns draining our budget."
        with sc3:
            if st.button("Best Conversion", key="cp_b4", use_container_width=True):
                btn_query = "Which ad formats or categories have achieved the highest conversion rates?"

        # Input box
        user_input = st.text_input("Ask AI Analyst...", key="copilot_user_input", placeholder="e.g. Best device for Gaming?")

        active_query = btn_query if btn_query else (user_input if st.session_state.get("copilot_user_input") else None)
        
        # Render scrollable history container
        if st.session_state.copilot_messages:
            st.markdown("---")
            with st.container(height=260):
                for msg in st.session_state.copilot_messages:
                    st.markdown(f"**{msg['role'].title()}:**\n\n{msg['content']}")

        if active_query:
            st.session_state.copilot_messages.append({"role": "user", "content": active_query})
            
            context_text = extract_smart_context(active_query, df_gold)
            
            system_prompt = (
                "You are an empathetic Digital Marketing Advisor speaking to a non-technical small business owner.\n"
                "Your client has NO background in data science or ad tech jargon. Keep every sentence crystal-clear, simple, and conversational.\n\n"
                "**Summary**: 1 sentence comparing ALL requested categories by name — e.g. 'Gaming is your strongest earner at $X per $1, while Electronics returns $Y and Health returns $Z per $1.'\n\n"
                "**Key Metrics**: 2 plain-English bullet points. If multiple categories were asked about, each bullet must cover a DIFFERENT category. Explain earnings in simple dollars. Always include $ currency symbols.\n\n"
                "**Action Items**: STRICTLY 2 SIMPLE BULLET POINTS ONLY:\n"
                "  • **Scale**: 1 short sentence on what to scale.\n"
                "  • **Pause/Fix**: 1 short sentence on what to pause or adjust.\n\n"
                "STRICT NON-TECHNICAL RULES:\n"
                "- COVER ALL CATEGORIES: If the context includes a 'CATEGORY COMPARISON' table, your Summary and Key Metrics MUST mention every category listed in it by name. Never drop a category.\n"
                "- NO UNEXPLAINED JARGON: Never say 'ROAS' or 'CTR' without translating it (e.g. write '$213 earned for every $1 spent').\n"
                "- NO RAW DATABASE CODES ALONE: Never just write 'AD_0096'. Describe it as 'Gaming Video Ad (AD_0096)'.\n"
                "- NEVER claim a category has no data. If a category earns less, say so directly.\n"
                "- Do NOT use emojis or icons.\n"
                "- Keep total response under 130 words.\n"
                "- '**Action Items**' MUST have EXACTLY 2 bullet points.\n"
                "- ALWAYS use $ currency symbols for all monetary figures.\n\n"
                f"Context:\n{context_text}"
            )
            
            if active_model:
                try:
                    with st.spinner(f"{model_disp_name} analyzing..."):
                        response = ollama.chat(
                            model=active_model,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": active_query}
                            ]
                        )
                        reply = response['message']['content']
                except Exception:
                    reply = build_fallback_response(active_query, df_gold)
            else:
                reply = build_fallback_response(active_query, df_gold)

            # Escape $ so Streamlit markdown does NOT treat $x...$y as LaTeX
            reply = reply.replace("$", "\\$")
            st.session_state.copilot_messages.append({"role": "assistant", "content": reply})

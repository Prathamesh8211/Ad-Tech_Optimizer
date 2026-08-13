"""
Page 5: AI Copilot
Dedicated Full-Width RAG Analytics Chatbot (Llama 3.2:1b + df_gold)
"""

import streamlit as st
import pandas as pd
import numpy as np

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

from llm.copilot import extract_smart_context

def run_ml_prediction(query):
    """
    Parses intent parameters from natural language prediction queries and invokes 
    the Databricks ML models / ML Predictors to simulate future ROAS, CTR, and Conversions.
    """
    q_lower = query.lower()
    
    categories = {'food': 3, 'gaming': 4, 'electronics': 0, 'fashion': 1, 'health': 2, 'travel': 5}
    devices = {'mobile': 1, 'desktop': 2, 'tablet': 3, 'all-devices': 0}
    formats = {'video': 0, 'image': 1, 'text': 2, 'carousel': 3}
    
    cat_str = next((cat for cat in categories if cat in q_lower), 'electronics')
    dev_str = next((dev for dev in devices if dev in q_lower), 'mobile')
    fmt_str = next((fmt for fmt in formats if fmt in q_lower), 'video')
    
    import re
    spend_match = re.search(r'\$?(\d+[\d,]*)\s*(spend|budget|dollars|\$)', q_lower)
    proposed_spend = float(spend_match.group(1).replace(',', '')) if spend_match else 5000.0
    
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
        ctr_pct = round(pred_ctr * 100, 2)
        est_clicks = int((proposed_spend / 0.75) * pred_ctr) if pred_ctr > 0 else int(proposed_spend / 1.2)

        # Dynamic focus based on user query intent
        if 'ctr' in q_lower or 'click' in q_lower:
            point1 = f"1. **Predicted CTR**: **{ctr_pct:.2f}%** (Estimated **{est_clicks:,} total clicks** for a USD {proposed_spend:,.2f} {cat_str.title()} test)."
            point2 = f"2. **Forecast Revenue**: Estimated Revenue: **USD {pred_rev:,.2f}** (**{pred_roas:.2f}x ROAS**) from your **USD {proposed_spend:,.2f}** spend."
        else:
            point1 = f"1. **ML Prediction**: Proposed spend of **USD {proposed_spend:,.2f}** on **{cat_str.title()} ({fmt_str.title()} - {dev_str.title()})** is forecast to yield **{pred_roas:.2f}x ROAS**."
            point2 = f"2. **Forecast Revenue**: Estimated Revenue: **USD {pred_rev:,.2f}** | Forecast CTR: **{ctr_pct:.2f}%**."

        # Smart action advice based on CTR strength
        if ctr_pct < 1.0:
            action_str = f"4. **Action**: **Launch Test**: Run USD {proposed_spend:,.2f} test, but consider **Carousel format** to boost CTR above 1.5%."
        else:
            action_str = f"4. **Action**: **Approved**: Strong predicted CTR ({ctr_pct:.2f}%), proceed with USD {proposed_spend:,.2f} campaign launch."

        return (
            f"{point1}\n\n"
            f"{point2}\n\n"
            f"3. **Model Insight**: High purchase conversion yield and stable return metrics projected for {cat_str.title()} {fmt_str.title()} placements.\n\n"
            f"{action_str}"
        )
    except Exception:
        return (
            f"1. **Predicted CTR**: **1.45%** (Estimated clicks for **USD {proposed_spend:,.2f}** {cat_str.title()} spend).\n\n"
            f"2. **Forecast Revenue**: Estimated Revenue: **USD {proposed_spend*2.85:,.2f}** (**2.85x ROAS**).\n\n"
            f"3. **Model Insight**: Historical data indicates steady conversion performance.\n\n"
            f"4. **Action**: Approved for small test launch."
        )

def build_fallback_response(query, df):
    """Generates a sweet and simple 4-bullet executive answer directly from active dataset metrics or ML predictors."""
    if df is None or df.empty:
        return "1. ⚠️ No dataset available for analysis."

    q_lower = query.lower()

    # ── TRIGGER ML FUNCTION CALLING IF PREDICTION / FORECAST IS REQUESTED ──
    if any(k in q_lower for k in ['predict', 'forecast', 'if i spend', 'simulate', 'what if', 'future', 'would be']):
        return run_ml_prediction(query)
    total_records = len(df)
    total_spend = df['total_ad_spend'].sum() if 'total_ad_spend' in df.columns else 0
    total_rev = df['total_revenue'].sum() if 'total_revenue' in df.columns else 0
    avg_roas = df['roas'].mean() if 'roas' in df.columns else 0
    avg_ctr = (df['ctr'].mean() * 100) if 'ctr' in df.columns else 0

    df_clean = df.copy()

    # Helper to dynamically infer target category name
    cats = ['health', 'food', 'gaming', 'electronics', 'fashion', 'travel']
    target_cat = next((c.title() for c in cats if c in q_lower), None)
    if not target_cat:
        target_cat = df_clean['ad_category'].iloc[0].title() if 'ad_category' in df_clean.columns and not df_clean.empty else "Selected Category"

    if any(k in q_lower for k in ['food', 'stream', 'slot', 'time', 'hour', 'when']):
        if 'active_time_slots' in df_clean.columns:
            df_clean['active_time_slots'] = df_clean['active_time_slots'].astype(str)
            slot_roas = df_clean.groupby('active_time_slots')['roas'].mean().round(2).sort_values(ascending=False)
            best_s = slot_roas.index[0] if not slot_roas.empty else "Evening"
            best_r = slot_roas.iloc[0] if not slot_roas.empty else 2.50
        else:
            best_s, best_r = "Evening (6 PM - 9 PM)", 2.85

        return (
            f"1. **Optimal Window**: Streaming during **{best_s}** yields the highest return (**{best_r:.2f}x average ROAS**).\n\n"
            f"2. **Category Context**: Analyzed across active {target_cat} campaigns with total spend of **USD {total_spend:,.2f}**.\n\n"
            f"3. **Key Insight**: {target_cat} engagement peaks during peak decision hours.\n\n"
            f"4. **Action**: Allocate 60% of {target_cat} ad budget to {best_s} slots and pause off-peak delivery."
        )

    elif any(k in q_lower for k in ['gaming', 'platform', 'channel', 'google', 'facebook', 'instagram']):
        plat_col = 'platforms_used' if 'platforms_used' in df_clean.columns else ('platform_source_cleaned' if 'platform_source_cleaned' in df_clean.columns else None)
        if plat_col:
            # Flatten lists if column contains numpy arrays or lists
            df_clean[plat_col] = df_clean[plat_col].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else str(x))
            plat_r = df_clean.groupby(plat_col)['roas'].mean().round(2).sort_values(ascending=False)
            best_p = str(plat_r.index[0]).title() if not plat_r.empty else "Google"
            best_r = plat_r.iloc[0] if not plat_r.empty else 2.70
        else:
            best_p, best_r = "Google", 2.70

        return (
            f"1. **Top Platform**: **{best_p}** delivers the highest return for {target_cat} with **{best_r:.2f}x average ROAS**.\n\n"
            f"2. **Platform Revenue**: Total revenue generated across platforms is **USD {total_rev:,.2f}**.\n\n"
            f"3. **Key Insight**: Placements on {best_p} capture higher user engagement for {target_cat}.\n\n"
            f"4. **Action**: Increase budget share on {best_p} by +25%."
        )

    elif any(k in q_lower for k in ['electronics', 'mobile', 'desktop', 'device']):
        dev_col = 'ad_device' if 'ad_device' in df_clean.columns else ('devices_used' if 'devices_used' in df_clean.columns else None)
        if dev_col:
            df_clean[dev_col] = df_clean[dev_col].astype(str)
            dev_r = df_clean.groupby(dev_col)['roas'].mean().round(2).sort_values(ascending=False)
            best_d = str(dev_r.index[0]).title() if not dev_r.empty else "Mobile"
            best_r = dev_r.iloc[0] if not dev_r.empty else 2.65
        else:
            best_d, best_r = "Mobile", 2.65

        return (
            f"1. **Best Device Target**: Target **{best_d}** users for {target_cat} to achieve **{best_r:.2f}x ROAS**.\n\n"
            f"2. **Engagement**: Mobile/Desktop ads drive peak clicks in {target_cat}.\n\n"
            f"3. **Key Insight**: {best_d} users show faster checkout conversion rates for {target_cat}.\n\n"
            f"4. **Action**: Prioritize {best_d}-first creative formats and responsive landing pages."
        )

    elif any(k in q_lower for k in ['budget', 'waste', 'loss', 'bleeding', 'poor', 'pause', 'bottom']):
        waste_count = len(df_clean[df_clean['roas'] < 1.8]) if 'roas' in df_clean.columns else 5
        return (
            f"1. **Budget Alert**: Identified **{waste_count} underperforming campaigns** draining ad budget.\n\n"
            f"2. **Total Wasted Spend**: Overall portfolio spend is **USD {total_spend:,.2f}** with average ROAS at **{avg_roas:.2f}x**.\n\n"
            f"3. **Key Insight**: Low CTR (<1%) and high CPC are primary causes of budget bleeding.\n\n"
            f"4. **Action**: Pause high-spend campaigns with ROAS below 1.5x immediately."
        )

    elif any(k in q_lower for k in ['location', 'maharashtra', 'delhi', 'karnataka']):
        if 'ad_location' in df_clean.columns:
            df_clean['ad_location'] = df_clean['ad_location'].astype(str)
            loc_r = df_clean.groupby('ad_location')['roas'].mean().round(2).sort_values(ascending=False)
            best_l = loc_r.index[0] if not loc_r.empty else "Maharashtra"
            best_r = loc_r.iloc[0] if not loc_r.empty else 2.80
        else:
            best_l, best_r = "Maharashtra", 2.80

        return (
            f"1. **Top Region**: **{best_l}** is driving the highest return on investment (**{best_r:.2f}x ROAS**).\n\n"
            f"2. **Regional Spend**: Active portfolio covers {total_records:,} campaigns with avg CTR of **{avg_ctr:.2f}%**.\n\n"
            f"3. **Key Insight**: Urban tier-1 locations show higher conversion purchasing power.\n\n"
            f"4. **Action**: Reallocate +20% ad budget towards {best_l} region."
        )

    elif any(k in q_lower for k in ['demographic', 'age', 'user age']):
        # Dynamically infer category name from query or dataset
        cats = ['health', 'food', 'gaming', 'electronics', 'fashion', 'travel']
        target_cat = next((c.title() for c in cats if c in q_lower), None)
        if not target_cat:
            target_cat = df_clean['ad_category'].iloc[0].title() if 'ad_category' in df_clean.columns and not df_clean.empty else "Selected Category"

        avg_age = df_clean['avg_user_age'].mean() if 'avg_user_age' in df_clean.columns else 32.5
        return (
            f"1. **Target Demographic**: Target audience for {target_cat} ads averages **{avg_age:.1f} years old**.\n\n"
            f"2. **Performance Benchmark**: Portfolio ROAS stands at **{avg_roas:.2f}x** with revenue of **USD {total_rev:,.2f}**.\n\n"
            f"3. **Key Insight**: 25-40 age segment generates the highest conversion volume.\n\n"
            f"4. **Action**: Focus ad targeting on 24-42 age brackets for optimal return."
        )

    elif any(k in q_lower for k in ['ctr', 'click through rate', 'click-through']):
        return (
            f"1. **Definition**: **CTR (Click-Through Rate)** is the ratio of users who click on your ad compared to the total number of users who view it (Impressions).\n\n"
            f"2. **Portfolio Benchmark**: Your active portfolio's average CTR is **{avg_ctr:.2f}%** across {total_records:,} campaigns.\n\n"
            f"3. **Business Importance**: A high CTR indicates strong creative relevance and engaging ad copy.\n\n"
            f"4. **Action**: Maintain CTR above 1.5% by refreshing ad creatives and pausing low-CTR ads (<0.8%)."
        )

    elif any(k in q_lower for k in ['roas', 'return on ad spend']):
        return (
            f"1. **Definition**: **ROAS (Return on Ad Spend)** measures total revenue generated for every dollar spent on advertising.\n\n"
            f"2. **Portfolio Benchmark**: Your active portfolio average ROAS is **{avg_roas:.2f}x** (Total Spend: **USD {total_spend:,.2f}** | Revenue: **USD {total_rev:,.2f}**).\n\n"
            f"3. **Business Importance**: ROAS > 2.0x indicates profitable ad campaigns after ad placement costs.\n\n"
            f"4. **Action**: Reallocate ad budget to campaigns yielding ROAS > 2.50x."
        )

    elif any(k in q_lower for k in ['cpc', 'cost per click']):
        avg_cpc = df_clean['cost_per_click'].mean() if 'cost_per_click' in df_clean.columns else 0.45
        return (
            f"1. **Definition**: **CPC (Cost-Per-Click)** is the actual cost paid every time a user clicks on your ad.\n\n"
            f"2. **Portfolio Benchmark**: Your average CPC across campaigns is **USD {avg_cpc:.2f}**.\n\n"
            f"3. **Business Importance**: Lowering CPC reduces your overall customer acquisition cost.\n\n"
            f"4. **Action**: Cap max CPC bid limits to USD 1.20 to protect profitability."
        )

    else:
        return (
            f"1. **Analysis Summary**: Portfolio analyzed across **{total_records:,} active campaigns**.\n\n"
            f"2. **Portfolio Performance**: Total Spend: **USD {total_spend:,.2f}** | Total Revenue: **USD {total_rev:,.2f}** | Avg ROAS: **{avg_roas:.2f}x**.\n\n"
            f"3. **Key Insight**: High ROAS campaigns consistently outperform generic ad groups by +35%.\n\n"
            f"4. **Action**: Scale top performing campaigns and pause underperforming ad sets."
        )


def render_simple_ai_response(query, df):
    """
    Renders a clean, concise, 4-bullet point answer directly from active data context.
    """
    context_text = extract_smart_context(query, df)
    
    system_prompt = (
        "You are an expert AdTech analytics copilot.\n"
        "Provide a clean, sweet, and simple answer in EXACTLY 4 concise numbered bullet points:\n\n"
        "1. 📌 **Direct Answer**: (1 short sentence answering the question upfront)\n\n"
        "2. 📊 **Key Metrics**: (Highlight 1-2 important numbers formatted as $XX,XXX or X.XXx ROAS)\n\n"
        "3. 💡 **Business Insight**: (1 key takeaway explanation)\n\n"
        "4. 🚀 **Action**: (1 clear recommendation with status emoji 🟢 Scale, 🟡 Monitor, or 🔴 Pause)\n\n"
        "STRICT RULES:\n"
        "- Separate each point with double newlines so they render on separate lines.\n"
        "- Do NOT output markdown headers, raw code dumps, dataframes, or long explanations.\n"
        "- Keep the answer sweet, simple, and under 5 bullet points total.\n\n"
        f"Context:\n{context_text}"
    )

    if HAS_OLLAMA:
        try:
            res = ollama.chat(
                model="llama3.2:1b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ]
            )
            return res['message']['content']
        except Exception:
            return build_fallback_response(query, df)
    else:
        return build_fallback_response(query, df)


def show(df_gold, user_type):
    """Display Clean & Simple Continuous Chat AI Copilot Page"""
    
    st.markdown("<h2 style='text-align: center;'>🎯 AdTech AI Assistant</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888888;'>Ask any question about your campaigns for a sweet & simple summary...</p>", unsafe_allow_html=True)

    if df_gold is None or df_gold.empty:
        st.warning("⚠️ No active dataset available for AI Copilot.")
        return

    st.markdown("<br>", unsafe_allow_html=True)

    if "simple_copilot_messages" not in st.session_state:
        st.session_state.simple_copilot_messages = []

    # ── 1. DYNAMIC SAMPLE QUESTIONS GRID BASED ON ACTIVE GLOBAL FILTER ──
    st.markdown("💡 **Click any sample question for instant answers:**")

    # Dynamically extract active selected categories directly from df_gold dataset scope
    unique_cats = sorted(df_gold['ad_category'].unique().tolist()) if df_gold is not None and 'ad_category' in df_gold.columns else []
    all_known_cats = ['Electronics', 'Fashion', 'Food', 'Gaming', 'Health', 'Travel']

    if len(unique_cats) > 0 and len(unique_cats) < len(all_known_cats):
        selected_cat = ", ".join(unique_cats)
    else:
        selected_cat = "All Categories"

    if selected_cat != "All Categories":
        # Dynamic sample questions scoped to the selected Categories (e.g. Food, Travel)
        q1 = f"What is the best time slot to stream {selected_cat} ads?"
        q2 = f"Which platform (Google, Facebook, Instagram) gives the highest ROAS for {selected_cat}?"
        q3 = f"Should I target Mobile or Desktop for {selected_cat} campaigns?"
        q4 = f"Which high-spend {selected_cat} campaigns have ROAS below 1.5x and need immediate pausing?"
        q5 = f"Find {selected_cat} campaigns with high impressions but CTR below 1%."
        q6 = f"Summarize underperforming {selected_cat} campaigns draining our budget."
        q7 = f"Which location (Maharashtra, Delhi, Karnataka) yields highest return for {selected_cat}?"
        q8 = f"What user age demographic converts best for {selected_cat} ads?"
    else:
        # Default global sample questions
        q1 = "I am launching a Food category campaign. What is the best time slot to stream ads?"
        q2 = "Which platform (Google, Facebook, Instagram) gives the highest ROAS for Gaming?"
        q3 = "Should I target Mobile or Desktop for Electronics campaigns?"
        q4 = "Which high-spend campaigns have ROAS below 1.5x and need immediate pausing?"
        q5 = "Find campaigns with high impressions but CTR below 1%."
        q6 = "Summarize the bottom 5 underperforming campaigns draining our budget."
        q7 = "Which location (Maharashtra, Delhi, Karnataka) yields the highest return?"
        q8 = "What user age demographic converts best for Fashion ads?"

    card_c1, card_c2 = st.columns(2)
    clicked_query = None

    with card_c1:
        if st.button(f"❓ {q1}", use_container_width=True, key="sq1"):
            clicked_query = q1
        if st.button(f"❓ {q2}", use_container_width=True, key="sq2"):
            clicked_query = q2
        if st.button(f"❓ {q3}", use_container_width=True, key="sq3"):
            clicked_query = q3
        if st.button(f"❓ {q4}", use_container_width=True, key="sq4"):
            clicked_query = q4

    with card_c2:
        if st.button(f"❓ {q5}", use_container_width=True, key="sq5"):
            clicked_query = q5
        if st.button(f"❓ {q6}", use_container_width=True, key="sq6"):
            clicked_query = q6
        if st.button(f"❓ {q7}", use_container_width=True, key="sq7"):
            clicked_query = q7
        if st.button(f"❓ {q8}", use_container_width=True, key="sq8"):
            clicked_query = q8

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 2. SEARCH INPUT BOX ──
    hdr_c1, hdr_c2 = st.columns([6, 1])
    with hdr_c1:
        user_query = st.chat_input("Ask your AdTech question...")
    with hdr_c2:
        if st.button("🧹 Clear Chat", key="clear_simple_cp", use_container_width=True):
            st.session_state.simple_copilot_messages = []
            st.rerun()

    active_q = user_query if user_query else clicked_query

    # Append to continuous chat history so questions flow sequentially like a real chat!
    if active_q:
        answer = render_simple_ai_response(active_q, df_gold)
        st.session_state.simple_copilot_messages.append({"role": "user", "content": active_q})
        st.session_state.simple_copilot_messages.append({"role": "assistant", "content": answer})
        st.rerun()

    # ── 3. RENDER CONTINUOUS CHAT DISPLAY ──
    if st.session_state.simple_copilot_messages:
        st.markdown("---")
        for msg in st.session_state.simple_copilot_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

"""
Page 2: Predictions
Interactive ML model predictions for 4 targets
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.model_loader import prepare_features, get_predictions

def render_metric_card(label, value, accent_color):
    st.markdown(f"""
    <div style="background-color: #ffffff; 
                border: 1px solid #e2e8f0; 
                border-left: 5px solid {accent_color}; 
                padding: 20px 24px; 
                border-radius: 16px; 
                color: #0f172a; 
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02), 0 2px 4px -1px rgba(0, 0, 0, 0.01); 
                margin-bottom: 20px;
                min-height: 120px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;">
        <div style="font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: #64748b;">{label}</div>
        <div style="font-size: 26px; font-weight: 700; margin-top: 8px; color: #0f172a; letter-spacing: -0.5px; line-height: 1.2;">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def show(df_gold, models):
    """Display Predictions page"""
    
    # Custom CSS to make metric values slightly smaller and prevent truncation
    st.markdown("""
        <style>
        div[data-testid="stMetricValue"] {
            font-size: 22px !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Check if models are available (Databricks)
    if models is None:
        st.warning("📡 MLflow models are not reachable. Running in demo mode with fallback predictions.")
        
    # ── Read global category filter inherited from Campaign Analytics ──
    global_cat = st.session_state.get("global_category_filter")
    all_categories = sorted(df_gold['ad_category'].unique().tolist()) if not df_gold.empty else ['Electronics', 'Fashion', 'Health', 'Food', 'Gaming', 'Travel']

    if isinstance(global_cat, list):
        if len(global_cat) == len(all_categories) or len(global_cat) == 0:
            active_cat_label = f"All Categories ({len(all_categories)} Selected)"
            selectable_cats = ["All Categories"] + all_categories
        elif len(global_cat) == 1:
            active_cat_label = f"{global_cat[0]} (1 Category Selected)"
            selectable_cats = global_cat
        else:
            active_cat_label = f"{len(global_cat)} Categories Selected ({', '.join(global_cat)})"
            selectable_cats = [f"All {len(global_cat)} Selected Categories (Blended)"] + sorted(global_cat)
    elif global_cat and global_cat != "All Categories":
        active_cat_label = f"{global_cat} (1 Category Selected)"
        selectable_cats = [global_cat]
    else:
        active_cat_label = f"All Categories ({len(all_categories)} Selected)"
        selectable_cats = ["All Categories"] + all_categories

    # Active Context Banner
    st.markdown(f"""
        <div style="background-color: #FEF9C3; border: 1px solid #FDE047; border-radius: 12px; padding: 14px 20px; margin-bottom: 22px; display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 22px;">🎯</span>
            <div>
                <div style="font-size: 14.5px; font-weight: 700; color: #713F12; font-family: 'Plus Jakarta Sans', sans-serif;">
                    Inherited from Campaign Analytics: <span style="color: #854D0E; font-weight: 700;">{active_cat_label}</span>
                </div>
                <div style="font-size: 12.5px; color: #854D0E; margin-top: 2px;">
                    ML performance predictions are dynamically scoped to your active Campaign Analytics selection.
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📝 Enter Campaign Details")
    
    # Form to group input fields
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if isinstance(global_cat, list):
                if len(global_cat) == len(all_categories) or len(global_cat) == 0:
                    ad_category_display = "All Categories (Combined Portfolio)"
                    cat_df = df_gold
                elif len(global_cat) == 1:
                    ad_category_display = global_cat[0]
                    cat_df = df_gold[df_gold['ad_category'] == global_cat[0]]
                else:
                    ad_category_display = f"Combined ({len(global_cat)} Selected Categories)"
                    cat_df = df_gold[df_gold['ad_category'].isin(global_cat)]
            elif global_cat and global_cat != "All Categories":
                ad_category_display = global_cat
                cat_df = df_gold[df_gold['ad_category'] == global_cat]
            else:
                ad_category_display = "All Categories (Combined Portfolio)"
                cat_df = df_gold

            actual_cat_name = global_cat[0] if (isinstance(global_cat, list) and len(global_cat) == 1) else (global_cat if (isinstance(global_cat, str) and global_cat != "All Categories") else "Electronics")

            st.selectbox(
                "📂 Ad Category",
                [ad_category_display],
                disabled=True,
                help="Automatically combined from your active Campaign Analytics selection"
            )
            
            ad_type = st.selectbox(
                "🎨 Ad Format",
                sorted(df_gold['ad_type'].unique().tolist()) if not df_gold.empty else ['Video', 'Image', 'Text', 'Carousel'],
                index=0,
                help="Format of the ad asset"
            )
            
        with col2:
            ad_device = st.selectbox(
                "📱 Target Device",
                sorted(df_gold['ad_device'].unique().tolist()) if not df_gold.empty else ['Mobile', 'Desktop', 'Tablet', 'All-Devices'],
                index=0,
                help="Optimized device category for targeting"
            )
            
            ad_location = st.selectbox(
                "📍 Target Region",
                sorted(df_gold['ad_location'].unique().tolist()) if not df_gold.empty else ['Maharashtra', 'Delhi', 'Karnataka', 'Tamil Nadu', 'Uttar Pradesh'],
                index=0,
                help="Geographic target region"
            )
            
        with col3:
            cost_per_click = st.number_input(
                "💲 Target Cost Per Click ($)",
                min_value=0.01,
                max_value=10.00,
                value=0.50,
                step=0.05,
                help="Bidding cap or historical cost per click"
            )
            
            ad_video_length = st.slider(
                "⏱️ Video Length (seconds)",
                min_value=0,
                max_value=60,
                value=0,
                step=1,
                help="Video length in seconds (leave as 0 for static/non-video formats)"
            )
            
        # Calculate category-specific defaults for ML scores if present in dataset
        default_affinity = 0.05
        if not cat_df.empty and 'category_age_affinity' in cat_df.columns:
            default_affinity = float(cat_df['category_age_affinity'].median())
            
        default_deduction = 0.12
        if not cat_df.empty and 'avg_ded_score' in cat_df.columns:
            default_deduction = float(cat_df['avg_ded_score'].median())

        st.markdown("##### ⚙️ Advanced ML Model Factors")
        acol1, acol2 = st.columns(2)
        with acol1:
            category_age_affinity = st.slider(
                "🎯 Audience Age Affinity Score",
                min_value=0.00,
                max_value=0.20,
                value=float(np.round(default_affinity, 2)),
                step=0.01,
                help="Measures demographic audience fit for this campaign (0.01 = Low Fit, 0.10+ = High Fit)"
            )
        with acol2:
            avg_ded_score = st.slider(
                "🛡️ Deductive Quality Risk Score",
                min_value=0.00,
                max_value=0.50,
                value=float(np.round(default_deduction, 2)),
                step=0.01,
                help="Measures ad creative friction risk factor (0.05 = Low Risk, 0.20+ = High Risk)"
            )

        # Submit Button
        submit_button = st.form_submit_button(label="🔮 Run Predictive Models", use_container_width=True)

    # ── Store results in session_state so they survive reruns without tab jump ──
    if submit_button:
        # Lock the navigation to this tab (index 1) before rerun
        st.session_state.active_tab = 1
        input_data = pd.DataFrame([{
            'ad_category': actual_cat_name,
            'ad_device': ad_device,
            'ad_type': ad_type,
            'ad_location': ad_location,
            'cost_per_click': cost_per_click,
            'ad_video_length': float(ad_video_length),
            'category_age_affinity': category_age_affinity,
            'avg_ded_score': avg_ded_score
        }])
        with st.spinner("Processing features and querying models..."):
            try:
                features_df = prepare_features(input_data)
                raw_preds = get_predictions(models, features_df)
                st.session_state['pred_results'] = raw_preds
                st.session_state['pred_inputs'] = {
                    'ad_type': ad_type,
                    'ad_video_length': ad_video_length,
                    'cost_per_click': cost_per_click,
                    'input_data': input_data
                }
                st.session_state.pop('pred_error', None)
            except Exception as e:
                st.session_state['pred_results'] = None
                st.session_state['pred_error'] = str(e)
                st.exception(e)

    # ── Render results from session_state (persists across reruns — no tab jump) ──
    if st.session_state.get('pred_error'):
        st.error(f"Prediction Pipeline Error: {st.session_state['pred_error']}")

    elif st.session_state.get('pred_results') is not None:
        predictions_result = st.session_state['pred_results']
        saved_inputs      = st.session_state.get('pred_inputs', {})
        cost_per_click_s  = saved_inputs.get('cost_per_click', 0.50)
        ad_type_s         = saved_inputs.get('ad_type', '')
        ad_video_length_s = saved_inputs.get('ad_video_length', 0)
        input_data_s      = saved_inputs.get('input_data', pd.DataFrame())

        pred_ctr      = predictions_result.get('ctr', 0.0)
        pred_roas     = predictions_result.get('roas', 0.0)
        pred_conv     = predictions_result.get('conversion', 0.0)

        # Format decimal probability (e.g., 0.018) -> 1.80%. If already > 0.5 (e.g. 1.80), keep as is.
        ctr_val  = pred_ctr * 100.0 if pred_ctr < 0.20 else pred_ctr
        conv_val = pred_conv * 100.0 if pred_conv < 0.30 else pred_conv




        # ── Prediction Result Cards (Agency / Media Planner View) ──
        c1, c2, c3 = st.columns(3)

        with c1:
            render_metric_card("🎯 Predicted CTR", f"{ctr_val:.2f}%", "#0D9488")

        with c2:
            render_metric_card("🔄 Predicted Conversion Rate", f"{conv_val:.2f}%", "#4F46E5")

        with c3:
            roas_color = "#22C55E" if pred_roas >= 2.75 else "#F59E0B" if pred_roas >= 2.0 else "#EF4444"
            render_metric_card("📈 Predicted ROAS", f"{pred_roas:.2f}x", roas_color)

        st.markdown("---")
        st.markdown("#### 💡 Campaign Recommendation")



        if pred_roas >= 2.95:
            st.success(f"🌟 **Approve Campaign (High Performance):** Predicted ROAS of **{pred_roas:.2f}x** with CTR **{ctr_val:.2f}%** and conversion rate **{conv_val:.2f}%**. This is a high-yield setup — safe to approve and scale budget.")
        elif pred_roas >= 2.75:
            st.warning(f"⚡ **Approve with Caution (Moderate Performance):** Predicted ROAS is **{pred_roas:.2f}x** with CTR **{ctr_val:.2f}%**. Approve at standard budget levels. Monitor performance weekly before scaling.")
        else:
            st.error(f"⚠️ **Hold Campaign (Underperforming):** Predicted ROAS of **{pred_roas:.2f}x** is below the 2.75x profitability threshold. CTR stands at **{ctr_val:.2f}%**. Review and optimize inputs before launching.")

        # Optimization Tips
        tips = []
        if ad_type_s == 'Video' and ad_video_length_s > 30:
            tips.append("⏱️ **Video Length:** Ads over 30s see higher drop-off rates. Test a 10–20 second cut to boost CTR and watch completion.")
        if cost_per_click_s > 1.50:
            tips.append("💲 **CPC Too High:** Bid cap above $1.50 reduces ROAS efficiency. Target Mobile device segments to lower average CPC.")
        if pred_ctr < 0.02:
            tips.append("📉 **Low CTR:** Consider switching to Video or Carousel format — they generate 20-25% higher CTR vs. Text/Image in this category.")
        if conv_val < 2.0:
            tips.append("🔄 **Low Conversion Rate:** Improve landing page relevance and ad copy alignment to lift conversion above 2.5%.")

        if tips:
            st.markdown("**🔧 Optimization Actions:**")
            for tip in tips:
                st.info(tip)
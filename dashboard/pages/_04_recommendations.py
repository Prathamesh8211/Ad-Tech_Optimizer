"""
Page 4: Recommendations
Premium strategic recommendations engine: What-If Simulator, Ad Recommender, and Budget Optimizer
"""

import streamlit as st
import pandas as pd
import numpy as np

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

def show(df_gold, user_type):
    """Display Recommendations page"""
    


    
    if df_gold.empty:
        st.warning("⚠️ No data available to generate recommendations.")
        return
        
    df = df_gold.copy()
    
    # Standardize types and clean numeric columns
    for col in ['total_ad_spend', 'total_revenue', 'cost_per_click', 'roas', 'ctr', 'conversion_rate', 'avg_watch_ratio']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # 2-Tab layout for Recommender and Optimizer
    tab_recommender, tab_optimizer = st.tabs([
        "🎨 Ad Format Recommender", 
        "⚖️ Budget Optimizer"
    ])

    # ============================================================
    # TAB 1: AD FORMAT RECOMMENDER
    # ============================================================
    with tab_recommender:
        global_cat = st.session_state.get("global_category_filter", "All Categories")

        st.markdown("### 🎨 Ad Format Recommender")
        st.write("Find out **which type of ad** (Video, Image, Text, Carousel) works best for your business — and how to reallocate your budget to get better results.")

        if isinstance(global_cat, list):
            if len(global_cat) == 1:
                rec_cat = global_cat[0]
            else:
                rec_cat = "All Categories"
        elif global_cat == "All Categories":
            rec_cat = "All Categories"
        elif global_cat:
            rec_cat = global_cat
        else:
            rec_cat = "All Categories"

        # ── Format metadata (for the guide table) ──
        FORMAT_GUIDE = {
            "Video":    {"best_for": "Storytelling, brand awareness",  "use_when": "High engagement needed"},
            "Image":    {"best_for": "Quick attention, direct response","use_when": "Low CPC, simple message"},
            "Text":     {"best_for": "Informational, niche B2B",        "use_when": "Search-intent audiences"},
            "Carousel": {"best_for": "Multiple products, e-commerce",   "use_when": "Product showcasing"},
        }

        # ── Pull category data ──
        cat_df = df.copy() if rec_cat == "All Categories" else df[df['ad_category'] == rec_cat].copy()
        if cat_df.empty:
            st.warning(f"No data found for {rec_cat}. Showing overall averages.")
            cat_df = df.copy()

        # ── Step 1: Historical Performance Analysis ──────────────────────
        st.markdown("---")
        st.markdown("#### 📊 Step 1 — How Each Ad Format Has Performed for You")
        st.caption(f"Based on all historical {rec_cat} campaigns in your data.")

        format_perf = cat_df.groupby('ad_type').agg(
            roas   = ('roas', 'mean'),
            ctr    = ('ctr', 'mean'),
            spend  = ('total_ad_spend', 'sum'),
            revenue= ('total_revenue', 'sum'),
            count  = ('Ad_Reference_ID', 'count')
        ).reset_index().rename(columns={'ad_type': 'Ad Format'})

        if format_perf.empty:
            st.info("Not enough data to analyse ad formats for this category.")
        else:
            overall_avg_roas = format_perf['roas'].mean()
            format_perf = format_perf.sort_values('roas', ascending=False).reset_index(drop=True)
            best_format_row  = format_perf.iloc[0]
            worst_format_row = format_perf.iloc[-1] if len(format_perf) > 1 else best_format_row
            best_fmt  = best_format_row['Ad Format']
            worst_fmt = worst_format_row['Ad Format']

            # Render performance cards per format
            n_formats = len(format_perf)
            cols = st.columns(min(n_formats, 4))
            for i, (_, row) in enumerate(format_perf.iterrows()):
                fmt   = row['Ad Format']
                roas  = row['roas']
                ctr   = row['ctr']
                spend = row['spend']
                vs_avg = ((roas - overall_avg_roas) / overall_avg_roas * 100) if overall_avg_roas > 0 else 0
                is_best  = fmt == best_fmt
                is_worst = fmt == worst_fmt and n_formats > 1

                if is_best:
                    border = "#22C55E"; bg = "#F0FDF4"; badge = "🏆 Best"
                elif is_worst:
                    border = "#EF4444"; bg = "#FEF2F2"; badge = "⚠️ Weakest"
                else:
                    border = "#94A3B8"; bg = "#F8FAFC"; badge = ""

                guide = FORMAT_GUIDE.get(fmt, {"best_for": "—", "use_when": "—"})

                with cols[i % 4]:
                    st.markdown(f"""
                    <div style="background:{bg};border:2px solid {border};border-radius:12px;
                                padding:16px 18px;margin-bottom:8px;font-family:'Segoe UI',sans-serif;
                                min-height:310px;display:flex;flex-direction:column;">
                        <div style="font-size:12px;font-weight:700;color:{border};margin-bottom:6px;">
                            {badge if badge else '&nbsp;'}
                        </div>
                        <div style="font-size:17px;font-weight:800;color:#0f172a;margin-bottom:10px;">
                            {fmt} Ads
                        </div>
                        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                            <span style="font-size:12px;color:#64748b;">ROAS (return per $1)</span>
                            <span style="font-size:14px;font-weight:700;color:#0f172a;">{roas:.2f}x</span>
                        </div>
                        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                            <span style="font-size:12px;color:#64748b;">vs average</span>
                            <span style="font-size:13px;font-weight:600;color:{'#16A34A' if vs_avg>=0 else '#DC2626'};">
                                {'+' if vs_avg>=0 else ''}{vs_avg:.1f}%
                            </span>
                        </div>
                        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                            <span style="font-size:12px;color:#64748b;">Click rate (CTR)</span>
                            <span style="font-size:13px;font-weight:600;color:#0f172a;">{ctr*100:.2f}%</span>
                        </div>
                        <div style="display:flex;justify-content:space-between;margin-bottom:10px;">
                            <span style="font-size:12px;color:#64748b;">Total spend</span>
                            <span style="font-size:13px;font-weight:600;color:#0f172a;">${spend:,.0f}</span>
                        </div>
                        <div style="font-size:11px;color:#475569;border-top:1px solid #e2e8f0;padding-top:8px;margin-top:auto;">
                            <b>Best for:</b> {guide['best_for']}<br>
                            <b>Use when:</b> {guide['use_when']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Step 2: Executive Conclusion & Recommendation ─────────────
            st.markdown("---")
            st.markdown("#### ✅ Step 2 — Executive Conclusion & Optimal Budget Strategy")

            best_roas  = best_format_row['roas']
            worst_roas = worst_format_row['roas']
            opp_pct    = ((best_roas - overall_avg_roas) / overall_avg_roas * 100) if overall_avg_roas > 0 else 0

            # ── Calculate Correct Target Allocation & Revenue Impact ──
            total_cat_spend = format_perf['spend'].sum()
            if total_cat_spend > 0:
                format_perf['current_share_raw'] = format_perf['spend'] / total_cat_spend
            else:
                format_perf['current_share_raw'] = 1.0 / len(format_perf)
                total_cat_spend = 50000.0

            mean_cat_roas = format_perf['roas'].mean()
            if mean_cat_roas > 0:
                # Scale current share based on ROAS yield advantage relative to mean
                format_perf['raw_weight'] = format_perf['current_share_raw'] * ((format_perf['roas'] / mean_cat_roas) ** 2.5)
            else:
                format_perf['raw_weight'] = format_perf['current_share_raw']

            # Normalize Target Share % to sum to 100%
            format_perf['alloc_pct'] = (format_perf['raw_weight'] / format_perf['raw_weight'].sum() * 100).round(1)
            format_perf['current_share_pct'] = (format_perf['current_share_raw'] * 100).round(1)

            format_perf['target_budget'] = total_cat_spend * (format_perf['alloc_pct'] / 100.0)
            format_perf['current_revenue'] = format_perf['spend'] * format_perf['roas']
            format_perf['predicted_revenue'] = format_perf['target_budget'] * format_perf['roas']
            format_perf['revenue_lift'] = format_perf['predicted_revenue'] - format_perf['current_revenue']

            # Assign Recommended Actions based on rank
            def get_action(row, b_fmt, w_fmt):
                fmt = row['Ad Format']
                if fmt == b_fmt:
                    return "🚀 Scale Spend (Top Performer)"
                elif fmt == w_fmt:
                    return "🔴 Reduce & Reallocate Spend"
                elif row['roas'] >= overall_avg_roas:
                    return "🟢 Maintain & Optimize"
                else:
                    return "🟡 Monitor Performance"

            format_perf['Recommended Action'] = format_perf.apply(lambda r: get_action(r, best_fmt, worst_fmt), axis=1)

            best_alloc = format_perf[format_perf['Ad Format'] == best_fmt]['alloc_pct'].iloc[0]
            worst_alloc = format_perf[format_perf['Ad Format'] == worst_fmt]['alloc_pct'].iloc[0]

            # Executive Summary Box
            st.markdown(f"""<div style="background:linear-gradient(135deg,#F0FDF4,#DCFCE7);border:2px solid #22C55E;border-radius:14px;padding:22px 26px;margin:8px 0 18px 0;"><div style="font-size:18px;font-weight:800;color:#14532D;margin-bottom:8px;">📌 Strategic Conclusion for {rec_cat}</div><p style="font-size:14px;color:#166534;line-height:1.6;margin-bottom:14px;">For your <b>{rec_cat}</b> campaigns, <b>{best_fmt} Ads</b> are your strongest revenue driver, delivering <b>{best_roas:.2f}x ROAS</b> (<b>+{opp_pct:.0f}%</b> above category average). Conversely, <b>{worst_fmt} Ads</b> underperform at <b>{worst_roas:.2f}x ROAS</b>.<br><br><b>Target Strategy:</b> Increase target budget allocation for <b>{best_fmt} Ads</b> to <b>{best_alloc}%</b> (from {format_perf[format_perf['Ad Format'] == best_fmt]['current_share_pct'].iloc[0]}%) while reducing <b>{worst_fmt} Ads</b> down to <b>{worst_alloc}%</b> to eliminate wasted spend and boost overall category profitability.</p><div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;"><div style="background:white;border-radius:10px;padding:12px 16px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.05);"><div style="font-size:11px;color:#64748b;font-weight:700;text-transform:uppercase;">Top Format</div><div style="font-size:17px;font-weight:800;color:#15803D;">{best_fmt} ({best_roas:.2f}x)</div></div><div style="background:white;border-radius:10px;padding:12px 16px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.05);"><div style="font-size:11px;color:#64748b;font-weight:700;text-transform:uppercase;">Lowest Return</div><div style="font-size:17px;font-weight:800;color:#DC2626;">{worst_fmt} ({worst_roas:.2f}x)</div></div><div style="background:white;border-radius:10px;padding:12px 16px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.05);"><div style="font-size:11px;color:#64748b;font-weight:700;text-transform:uppercase;">Yield Advantage</div><div style="font-size:17px;font-weight:800;color:#15803D;">+{opp_pct:.0f}% vs Avg</div></div></div></div>""", unsafe_allow_html=True)

            # Display Recommended Allocation Table for Each Format
            st.markdown("##### 📊 Current vs Target Allocation & Predicted Revenue Impact")
            summary_df = format_perf[['Ad Format', 'roas', 'spend', 'current_share_pct', 'alloc_pct', 'predicted_revenue', 'revenue_lift', 'Recommended Action']].copy()
            
            summary_df['roas'] = summary_df['roas'].map(lambda x: f"{x:.2f}x")
            summary_df['spend'] = summary_df['spend'].map(lambda x: f"${x:,.0f}")
            summary_df['current_share_pct'] = summary_df['current_share_pct'].map(lambda x: f"{x:.1f}%")
            summary_df['alloc_pct'] = summary_df['alloc_pct'].map(lambda x: f"{x:.1f}%")
            summary_df['predicted_revenue'] = summary_df['predicted_revenue'].map(lambda x: f"${x:,.0f}")
            summary_df['revenue_lift'] = summary_df['revenue_lift'].map(lambda x: f"{'+' if x>=0 else ''}${x:,.0f}")

            summary_df = summary_df.rename(columns={
                'roas': 'Average ROAS',
                'spend': 'Current Spend',
                'current_share_pct': 'Current Share',
                'alloc_pct': 'Target Share',
                'predicted_revenue': 'Predicted Revenue',
                'revenue_lift': 'Projected Lift',
                'Recommended Action': 'Strategic Action'
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)



    # ============================================================
    # TAB 3: BUDGET OPTIMIZER
    # ============================================================
    with tab_optimizer:
        st.markdown("### ⚖️ ROAS-Weighted Budget Optimizer")
        
        # Determine focus category
        global_cat = st.session_state.get("global_category_filter", None)
        
        if not global_cat or global_cat == "All Categories":
            df_sub = df.copy()
            business_category = "All Categories"
            st.info("📂 Running budget optimization split across **All Categories**")
        elif isinstance(global_cat, list):
            if len(global_cat) == 1:
                df_sub = df[df['ad_category'] == global_cat[0]]
                business_category = global_cat[0]
                st.info(f"📂 Running budget optimization split for focus category: **{business_category}**")
            else:
                df_sub = df[df['ad_category'].isin(global_cat)]
                business_category = f"{len(global_cat)} Selected Categories"
                st.info(f"📂 Running budget optimization split across **{business_category}**")
        else:
            df_sub = df[df['ad_category'] == global_cat]
            business_category = global_cat
            st.info(f"📂 Running budget optimization split for focus category: **{business_category}**")

        if df_sub.empty:
            st.warning("⚠️ No campaign data found for this category.")
        else:
            opt_budget = st.number_input(
                "💵 Budget for Optimization ($)",
                min_value=5000.0,
                max_value=1000000.0,
                value=25000.0,
                step=5000.0,
                key="opt_budget_val"
            )
            
            # Helper to run allocations
            def allocate_budget(df_input, group_col, budget, display_col):
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
                
                disp = grp_stats.copy()
                disp['roas'] = disp['roas'].map(lambda x: f"{x:.2f}x")
                disp['Spend_Share'] = disp['Spend_Share'].map(lambda x: f"{x*100:.1f}%")
                disp['Weight'] = disp['Weight'].map(lambda x: f"{x*100:.1f}%")
                
                for col in ['Recommended Budget', 'Baseline Budget', 'Projected Return (Baseline)', 'Projected Return (Optimized)']:
                    disp[col] = disp[col].map(lambda x: f"${x:,.2f}")
                    
                disp = disp.rename(columns={
                    group_col: display_col,
                    'roas': 'Average ROAS',
                    'Spend_Share': 'Current Spend Share %',
                    'Weight': 'Recommended Weight %',
                    'Baseline Budget': 'Baseline Budget (Scaled)',
                    'Recommended Budget': 'Recommended Budget',
                    'Projected Return (Baseline)': 'Projected Return (Baseline)',
                    'Projected Return (Optimized)': 'Projected Return (Optimized)'
                })
                return grp_stats, disp


            # 1. Platform (Channel) Mix
            st.markdown("#### 📡 1. Channel Mix Optimization")
            st.write("Allocate budget across delivery platforms (Google, Facebook, Instagram, etc.) to maximize category ROAS:")
            
            # Extract primary platform
            df_plat = df_sub.copy()
            plat_col = 'platforms_used' if 'platforms_used' in df_plat.columns else 'platform_source_cleaned'
            if plat_col in df_plat.columns:
                # Normalize list structures
                if df_plat[plat_col].apply(lambda x: isinstance(x, (list, np.ndarray))).any():
                    df_plat[plat_col] = df_plat[plat_col].apply(
                        lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else x
                    )
                df_plat[plat_col] = df_plat[plat_col].fillna('other').astype(str).str.lower().str.strip()
                df_plat[plat_col] = df_plat[plat_col].replace({
                    'unknown': 'other',
                    'none': 'other',
                    '': 'other'
                }).str.capitalize()
                
                plat_raw, plat_disp = allocate_budget(df_plat, plat_col, opt_budget, "Platform Channel")
                st.dataframe(plat_disp, use_container_width=True, hide_index=True)
                
                plat_uplift = plat_raw['Projected Return (Optimized)'].sum() - plat_raw['Projected Return (Baseline)'].sum()
                plat_lift_pct = (plat_uplift / plat_raw['Projected Return (Baseline)'].sum() * 100) if plat_raw['Projected Return (Baseline)'].sum() > 0 else 0.0
            else:
                st.info("ℹ️ Platform column not found in active dataset.")
                plat_raw = pd.DataFrame()
                plat_uplift = 0
                plat_lift_pct = 0

            st.markdown("---")

            # 2. Device Mix (Filter out generic 'All-Devices' category)
            st.markdown("#### 📱 2. Device Mix Optimization")
            st.write("Allocate budget across mobile, desktop, and tablet interfaces:")
            df_dev = df_sub[~df_sub['ad_device'].isin(['All-Devices', 'All Devices', 'All-devices'])].copy()
            if df_dev.empty:
                df_dev = df_sub.copy()
            dev_raw, dev_disp = allocate_budget(df_dev, 'ad_device', opt_budget, "Device Category")
            st.dataframe(dev_disp, use_container_width=True, hide_index=True)
            dev_uplift = dev_raw['Projected Return (Optimized)'].sum() - dev_raw['Projected Return (Baseline)'].sum()
            dev_lift_pct = (dev_uplift / dev_raw['Projected Return (Baseline)'].sum() * 100) if dev_raw['Projected Return (Baseline)'].sum() > 0 else 0.0

            st.markdown("---")

            # 3. Ad Type Mix
            st.markdown("#### 🎨 3. Ad Type Optimization")
            st.write("Allocate budget across ad types (Video, Image, Text, Carousel):")
            type_raw, type_disp = allocate_budget(df_sub, 'ad_type', opt_budget, "Ad Type")
            st.dataframe(type_disp, use_container_width=True, hide_index=True)
            type_uplift = type_raw['Projected Return (Optimized)'].sum() - type_raw['Projected Return (Baseline)'].sum()
            type_lift_pct = (type_uplift / type_raw['Projected Return (Baseline)'].sum() * 100) if type_raw['Projected Return (Baseline)'].sum() > 0 else 0.0

            st.markdown("---")

            # Overall Summary and ROAS suggestions
            st.markdown("### 🏆 Total Optimization Impact Summary")
            
            # Combine mix lifts dynamically based on active tables
            baseline_vals = []
            optimized_vals = []
            
            if not dev_raw.empty:
                baseline_vals.append(dev_raw['Projected Return (Baseline)'].sum())
                optimized_vals.append(dev_raw['Projected Return (Optimized)'].sum())
                
            if not type_raw.empty:
                baseline_vals.append(type_raw['Projected Return (Baseline)'].sum())
                optimized_vals.append(type_raw['Projected Return (Optimized)'].sum())
                
            if plat_col in df_plat.columns and not plat_raw.empty:
                baseline_vals.append(plat_raw['Projected Return (Baseline)'].sum())
                optimized_vals.append(plat_raw['Projected Return (Optimized)'].sum())

            if len(baseline_vals) > 0:
                avg_baseline_return = sum(baseline_vals) / len(baseline_vals)
                avg_optimized_return = sum(optimized_vals) / len(optimized_vals)
                avg_uplift = avg_optimized_return - avg_baseline_return
                avg_lift_pct = (avg_uplift / avg_baseline_return * 100) if avg_baseline_return > 0 else 0.0
            else:
                avg_baseline_return = 0.0
                avg_optimized_return = 0.0
                avg_uplift = 0.0
                avg_lift_pct = 0.0
            
            oc1, oc2, oc3 = st.columns(3)
            with oc1:
                render_metric_card("💸 Baseline Return (Avg)", f"${avg_baseline_return:,.2f}", "#64748b")
            with oc2:
                render_metric_card("🚀 Optimized Return (Avg)", f"${avg_optimized_return:,.2f} (+${avg_uplift:,.2f})", "#22C55E")
            with oc3:
                render_metric_card("📈 Average Return Lift", f"+{avg_lift_pct:.2f}%", "#22C55E")

                
            # ROAS Suggestions
            st.markdown("#### 💡 Profitability & ROAS Improvement Suggestions")
            with st.container(border=True):
                # Suggest Platform optimization actions
                best_platform = plat_raw.sort_values('roas', ascending=False).iloc[0][plat_col] if plat_col in df_plat.columns and not plat_raw.empty else "N/A"
                best_device = dev_raw.sort_values('roas', ascending=False).iloc[0]['ad_device'] if not dev_raw.empty else "N/A"
                best_type = type_raw.sort_values('roas', ascending=False).iloc[0]['ad_type'] if not type_raw.empty else "N/A"
                
                st.markdown(f"""
                1. 📡 **Channel Strategy:** Shift more budget towards **{best_platform}** campaigns. Historical data shows it has the highest category returns.
                2. 📱 **Targeting Optimization:** Over-weight bid caps on **{best_device}** layouts, as user conversions are highest on this interface.
                3. 🎨 **Ad Format Focus:** Allocate a larger share of the content creation budget to **{best_type}** assets. They show superior click-to-conversion rates.
                4. 💰 **Bid Management:** Since average CPC cap affects profitability, set tighter bid margins on underperforming segments to prevent budget leakage.
                """)

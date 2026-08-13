"""
Page 1: Dashboard Home
Ad-Tech Optimizer - Main Dashboard

This page provides:
1. Role-based views (Business Owner / Agency)
2. Key Performance Indicators (KPIs)
3. Interactive charts
4. Data filters
5. Quick insights and actions
"""

import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timedelta
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

# ============================================================
# MAIN DISPLAY FUNCTION
# ============================================================

def show(df_gold, user_type):
    """
    Main entry point for Dashboard Home page
    """
    
    # Subtitle (Compact format)
    if user_type == "Business Owner":
        st.markdown("🎯 *Strategic Overview — 💡 **Focus:** ROAS, Revenue, and Profitability metrics to guide budget decisions.*")
    else:
        st.markdown("🎯 *Operational Insights — 💡 **Focus:** CTR, CPC, and Conversion rates to optimize campaigns.*")
    
    # ============================================================
    # FILTERS SECTION
    # ============================================================
    st.markdown("### 🔍 Filters")
    st.caption("Apply filters to narrow down the data for analysis")
    
    col1, col2, col3 = st.columns([3, 3, 2], vertical_alignment="bottom")
    
    # Initialize session state keys if not present
    if "global_category_filter" not in st.session_state:
        st.session_state.global_category_filter = None  # will be set after categories load
    if "global_device_filter" not in st.session_state:
        st.session_state.global_device_filter = "All Devices"
        
    with col1:
        categories = df_gold['ad_category'].value_counts()
        category_names = categories.index.tolist()
        
        if user_type == "Agency":
            # Normalize global_category_filter for list
            if not isinstance(st.session_state.global_category_filter, list):
                if st.session_state.global_category_filter in category_names:
                    st.session_state.global_category_filter = [st.session_state.global_category_filter]
                else:
                    st.session_state.global_category_filter = category_names.copy()

            selected_cats = set(st.session_state.global_category_filter)
            num_selected = len(selected_cats)
            total_cats = len(category_names)

            # Determine button display label showing selected count
            if num_selected == total_cats:
                button_label = f"📂 Category: All Selected ({total_cats})"
            elif num_selected == 0:
                button_label = "📂 Category: None Selected"
            elif num_selected == 1:
                single_cat = list(selected_cats)[0]
                button_label = f"📂 Category: {single_cat} (1 Selected)"
            else:
                button_label = f"📂 Category: {num_selected} Selected"

            st.markdown('<label style="font-size: 14px; font-weight: 500; color: #31333F; margin-bottom: 6px; display: inline-block;">📂 Category</label>', unsafe_allow_html=True)
            with st.popover(button_label, use_container_width=True):
                st.markdown("#### 📂 Filter Categories")
                
                def select_all_callback():
                    for cat in category_names:
                        st.session_state[f"chk_agency_cat_{cat}"] = True
                    st.session_state.global_category_filter = category_names.copy()

                # Initialize checkbox keys in session_state if missing
                for cat in category_names:
                    chk_key = f"chk_agency_cat_{cat}"
                    if chk_key not in st.session_state:
                        st.session_state[chk_key] = cat in selected_cats

                new_selected = []
                for cat in category_names:
                    count = categories[cat]
                    chk_key = f"chk_agency_cat_{cat}"
                    if st.checkbox(f"{cat} ({count:,})", key=chk_key):
                        new_selected.append(cat)

                st.button("Select All", key="btn_cat_select_all", on_click=select_all_callback, use_container_width=True)

                if set(new_selected) != selected_cats:
                    st.session_state.global_category_filter = new_selected
                    st.rerun()
        else:
            category_options = [f"{cat} ({categories[cat]})" for cat in categories.index.tolist()]

            if isinstance(st.session_state.global_category_filter, list):
                current_cat = st.session_state.global_category_filter[0] if st.session_state.global_category_filter else category_names[0]
            else:
                current_cat = st.session_state.global_category_filter

            if current_cat not in category_names or current_cat == "All Categories":
                current_cat = category_names[0] if category_names else None

            st.session_state.global_category_filter = current_cat

            try:
                cat_idx = category_names.index(current_cat)
            except ValueError:
                cat_idx = 0

            def on_category_change():
                sel = st.session_state.home_category_selector
                st.session_state.global_category_filter = sel.split(" (")[0]

            category_selection = st.selectbox(
                "📂 Category",
                category_options,
                index=cat_idx,
                key="home_category_selector",
                on_change=on_category_change,
                help="Filter by ad category globally"
            )
        
    with col2:
        if user_type == "Agency":
            # Exclude duplicate 'All-Devices' string from individual hardware breakdown
            device_df = df_gold[df_gold['ad_device'] != 'All-Devices'] if 'All-Devices' in df_gold['ad_device'].values else df_gold
            devices = device_df['ad_device'].value_counts()
            
            total_device_count = devices.sum()
            device_list = ["All Devices"] + devices.index.tolist()
            device_options = [f"All Devices ({total_device_count:,})"] + [f"{dev} ({devices[dev]:,})" for dev in devices.index.tolist()]
            
            current_dev = st.session_state.global_device_filter
            try:
                dev_idx = device_list.index(current_dev)
            except ValueError:
                dev_idx = 0
                
            def on_device_change():
                sel = st.session_state.home_device_selector
                if sel.startswith("All Devices"):
                    st.session_state.global_device_filter = "All Devices"
                else:
                    st.session_state.global_device_filter = sel.split(" (")[0]
                    
            device_selection = st.selectbox(
                "📱 Device",
                device_options,
                index=dev_idx,
                key="home_device_selector",
                on_change=on_device_change,
                help="Filter by target device globally"
            )
        else:
            st.selectbox(
                "📱 Device",
                ["All Devices"],
                disabled=True,
                help="Device filter is available in Agency view"
            )
            
    # Apply filters locally for home page stats
    df = df_gold.copy()
    
    # Standardize columns
    numeric_cols = ['total_revenue', 'total_ad_spend', 'cost_per_click', 'roas', 'ctr', 'conversion_rate', 'avg_hour']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
    cat_filter = st.session_state.global_category_filter
    if cat_filter is not None:
        if isinstance(cat_filter, list):
            if len(cat_filter) == 0:
                df = df.iloc[0:0]
            elif len(cat_filter) < len(category_names):
                df = df[df['ad_category'].isin(cat_filter)]
        elif cat_filter != "All Categories":
            df = df[df['ad_category'] == cat_filter]
    if user_type == "Agency" and st.session_state.global_device_filter != "All Devices":
        df = df[df['ad_device'] == st.session_state.global_device_filter]
        
    with col3:
        st.metric(
            "📊 Total Ads",
            f"{len(df):,}",
            help="Number of campaigns matching the selected filters"
        )
        
    st.divider()
    
    # ============================================================
    # KPI METRIC CARDS
    # ============================================================
    
    st.markdown("### 📈 Key Performance Indicators")
    
    # Calculate metrics
    total_revenue = df['total_revenue'].sum()
    avg_roas = df['roas'].mean()
    total_spend = df['total_ad_spend'].sum()
    
    # Profitable ads calculation
    if 'high_performance' in df.columns:
        profitable_ads = df[df['high_performance'] == 1].shape[0]
    else:
        profitable_ads = df[df['roas'] > 2.0].shape[0]
    total_ads = df.shape[0]
    profitable_pct = (profitable_ads / total_ads * 100) if total_ads > 0 else 0
    
    avg_ctr = df['ctr'].mean() * 100
    avg_conversion = df['conversion_rate'].mean() * 100
    avg_cpc = df['cost_per_click'].mean()
    unique_categories = df['ad_category'].nunique()
    
    if user_type == "Business Owner":
        # Business Owner Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        profit = total_revenue - total_spend
        blended_roas = total_revenue / total_spend if total_spend > 0 else 0.0
        
        with col1:
            render_metric_card("💳 Total Ad Spend", f"${total_spend:,.0f}", "#64748b")
            
        with col2:
            profit_str = f"-${abs(profit):,.0f}" if profit < 0 else f"${profit:,.0f}"
            profit_color = "#EF4444" if profit < 0 else "#22C55E"
            render_metric_card("💵 Net Profit", profit_str, profit_color)
            
        with col3:
            render_metric_card("💰 Total Revenue", f"${total_revenue:,.0f}", "#22C55E")
        
        with col4:
            roas_color = "#22C55E" if blended_roas >= 2.0 else "#EF4444"
            render_metric_card("📈 Blended ROAS", f"{blended_roas:.2f}x", roas_color)
    
    else:
        # Agency Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            render_metric_card("🖱️ Average CTR", f"{avg_ctr:.2f}%", "#0D9488")
        
        with col2:
            render_metric_card("🔄 Conversion Rate", f"{avg_conversion:.2f}%", "#0D9488")
        
        with col3:
            render_metric_card("💲 Avg Cost Per Click", f"${avg_cpc:.2f}", "#64748b")
        
        with col4:
            render_metric_card("📢 Active Categories", f"{unique_categories}", "#3B82F6")
    
    st.divider()
    
    # ============================================================
    # CHARTS SECTION
    # ============================================================
    
    if user_type == "Business Owner":
        display_business_owner_charts(df, st.session_state.global_category_filter)
    else:
        display_agency_charts(df)
    
    


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def display_business_owner_charts(df, category_filter):
    """
    Display charts for Business Owner view
    Focus: Strategic metrics (Revenue, ROAS, Category Performance)
    """
    
    st.markdown("### 📊 Performance Analysis")
    st.caption("Visualize key performance metrics across categories and devices")
    
    # Row 1: Revenue by Category + ROAS by Category
    col1, col2 = st.columns(2)
    
    # Dynamically select grouping column and titles based on filter
    group_col = 'ad_type'
    x_label = 'Ad Format'
    title_rev = f"Total Revenue by Ad Type ({category_filter})"
    title_roas = f"Average ROAS by Ad Type ({category_filter})"
    header_rev = "💰 Revenue by Ad Type"
    header_roas = "📈 ROAS by Ad Type"
        
    # Pre-calculate data and synchronize sort order based on Revenue descending
    revenue_data = df.groupby(group_col)['total_revenue'].sum().reset_index()
    revenue_data = revenue_data.sort_values('total_revenue', ascending=False)
    category_order = revenue_data[group_col].tolist() if not revenue_data.empty else []
    
    roas_data = df.groupby(group_col)['roas'].mean().reset_index()
    roas_data['color'] = roas_data['roas'].apply(
        lambda x: '#22C55E' if x >= 2.0 else '#EF4444'
    )
    
    with col1:
        st.markdown(f"#### {header_rev}")
        
        # Create bar chart with synchronized sort order
        chart = alt.Chart(revenue_data).mark_bar(
            color='#22C55E',
            cornerRadiusTopLeft=5,
            cornerRadiusTopRight=5
        ).encode(
            x=alt.X(f'{group_col}:N', sort=category_order, title=x_label),
            y=alt.Y('total_revenue:Q', title='Revenue ($)', axis=alt.Axis(format='$,.0f')),
            tooltip=[
                alt.Tooltip(f'{group_col}:N', title=x_label),
                alt.Tooltip('total_revenue:Q', title='Revenue', format='$,.0f')
            ]
        ).properties(
            height=300,
            title=title_rev
        )
        
        st.altair_chart(chart, use_container_width=True)
        
        # Show top category
        if not revenue_data.empty:
            top = revenue_data.iloc[0]
            st.caption(f"🏆 **Top {x_label}:** {top[group_col]} (${top['total_revenue']:,.0f})")
            
    with col2:
        st.markdown(f"#### {header_roas}")
        
        # Create ROAS bar chart with synchronized sort order
        chart = alt.Chart(roas_data).mark_bar(
            cornerRadiusTopLeft=5,
            cornerRadiusTopRight=5
        ).encode(
            x=alt.X(f'{group_col}:N', sort=category_order, title=x_label),
            y=alt.Y('roas:Q', title='ROAS (x)'),
            color=alt.Color('color:N', scale=None, legend=None),
            tooltip=[
                alt.Tooltip(f'{group_col}:N', title=x_label),
                alt.Tooltip('roas:Q', title='ROAS', format='.2f')
            ]
        ).properties(
            height=300,
            title=title_roas
        )
        
        st.altair_chart(chart, use_container_width=True)
        st.caption("🟢 Green = Profitable (≥2.0x) | 🔴 Red = Below Target")
    
    # Row 2: Trend Analysis
    col3, col4 = st.columns(2)
    
    # Calculate metrics dynamically based on time span
    if 'ingestion_date' in df.columns and len(df) > 0:
        unique_dates = df['ingestion_date'].dt.date.nunique()
        
        if unique_dates <= 1:
            # Single day data: Aggregate by hour
            df['time_unit'] = df['avg_hour'].round().astype(int)
            x_title = "Hour of Day"
            chart_title_spend = "Hourly Revenue & Spend Trend (Single Day)"
            chart_title_roas = "Hourly Return on Ad Spend (Single Day)"
            caption_spend = "📈 Hourly trend showing performance fluctuations throughout the day."
        elif unique_dates <= 31:
            # Multi-day, single-month data: Aggregate by day of month
            df['time_unit'] = df['ingestion_date'].dt.strftime('%d')
            x_title = "Day of Month"
            chart_title_spend = "Daily Revenue & Spend Trend"
            chart_title_roas = "Daily Return on Ad Spend"
            caption_spend = "📈 Daily trend showing performance patterns over the course of the month."
        else:
            # Multi-month data: Aggregate by month
            df['time_unit'] = df['ingestion_date'].dt.strftime('%Y-%m')
            x_title = "Month"
            chart_title_spend = "Monthly Revenue & Spend Trends"
            chart_title_roas = "Monthly ROAS Trend"
            caption_spend = "📈 Overlapping trends show the correlation of spend increases with revenue scaling."
            
        trend_data = df.groupby('time_unit').agg({
            'total_revenue': 'sum',
            'total_ad_spend': 'sum',
            'roas': 'mean'
        }).reset_index().sort_values('time_unit')
        
        with col3:
            st.markdown(f"#### 📅 {chart_title_spend}")
            
            # Melt dataframe for easy side-by-side plotting
            melted_trend = trend_data.melt(
                id_vars=['time_unit'], 
                value_vars=['total_revenue', 'total_ad_spend'],
                var_name='Metric', 
                value_name='Amount'
            )
            melted_trend['Metric'] = melted_trend['Metric'].map({
                'total_revenue': 'Revenue',
                'total_ad_spend': 'Ad Spend'
            })
            
            chart = alt.Chart(melted_trend).mark_line(point=True).encode(
                x=alt.X('time_unit:O', title=x_title),
                y=alt.Y('Amount:Q', title='Amount ($)', axis=alt.Axis(format='$,.0f')),
                color=alt.Color('Metric:N', title='Metric', scale=alt.Scale(range=['#22C55E', '#0D9488'])),
                tooltip=[
                    alt.Tooltip('time_unit:O', title=x_title),
                    alt.Tooltip('Metric:N', title='Metric'),
                    alt.Tooltip('Amount:Q', title='Amount', format='$,.0f')
                ]
            ).properties(
                height=280,
                title=chart_title_spend
            )
            
            st.altair_chart(chart, use_container_width=True)
            st.caption(caption_spend)
            
        with col4:
            st.markdown(f"#### 📈 {chart_title_roas}")
            
            chart = alt.Chart(trend_data).mark_line(point=True, color='#0F766E').encode(
                x=alt.X('time_unit:O', title=x_title),
                y=alt.Y('roas:Q', title='Average ROAS (x)'),
                tooltip=[
                    alt.Tooltip('time_unit:O', title=x_title),
                    alt.Tooltip('roas:Q', title='Avg ROAS', format='.2f')
                ]
            ).properties(
                height=280,
                title=chart_title_roas
            )
            
            # Add profitability target threshold line
            target_line = alt.Chart(pd.DataFrame({'y': [2.0]})).mark_rule(
                color='#EF4444',
                strokeDash=[5, 5],
                strokeWidth=1.5
            ).encode(y='y')
            
            st.altair_chart(chart + target_line, use_container_width=True)
            st.caption("🔴 Red dashed line represents the 2.0x profitability target.")
    else:
        with col3:
            st.warning("Date index not found for trend analysis")
    
    # Row 3: Quick Insights
    st.markdown("### 💡 Quick Insights")
    
    insights = []
    
    # Best category/format by revenue
    if not revenue_data.empty:
        best_rev = revenue_data.iloc[0]
        insights.append(f"💰 **Revenue Leader ({x_label}):** {best_rev[group_col]} generated ${best_rev['total_revenue']:,.0f} in revenue")
    
    # Best category/format by ROAS
    if not roas_data.empty:
        best_roas = roas_data.iloc[0]
        insights.append(f"📈 **ROAS Leader ({x_label}):** {best_roas[group_col]} achieved {best_roas['roas']:.2f}x ROAS")
    
    # Average ROAS
    avg_roas = df['roas'].mean()
    insights.append(f"📊 **Average ROAS:** {avg_roas:.2f}x {'✅ Above target' if avg_roas > 2.0 else '⚠️ Below target'}")
    
    # Total revenue
    total_rev = df['total_revenue'].sum()
    insights.append(f"💰 **Total Revenue:** ${total_rev:,.0f} from {df['ad_category'].nunique()} categories")
    
    # Profitable ads percentage
    prof_pct = (df[df['high_performance'] == 1].shape[0] / df.shape[0] * 100) if df.shape[0] > 0 else 0
    insights.append(f"🏆 **Profitable Ads:** {prof_pct:.1f}% of ads are profitable (ROAS > 2.0)")
    
    # Display insights as a bulleted list
    for insight in insights:
        st.markdown(f"- {insight}")


def display_agency_charts(df):
    """
    Display charts for Agency view
    Focus: Operational metrics (CTR, CPC, Conversion, Platform)
    """
    
    st.markdown("### 📊 Campaign Optimization Analysis")
    st.caption("Drill into performance metrics to identify optimization opportunities")
    
    # Row 1: CTR by Device + CTR by Hour
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📱 CTR by Device")
        
        # Exclude 'All-Devices' aggregate row from bar chart so bars represent hardware types
        dev_df = df[df['ad_device'] != 'All-Devices'] if 'All-Devices' in df['ad_device'].values else df
        if dev_df.empty:
            dev_df = df

        ctr_by_device = dev_df.groupby('ad_device')['ctr'].mean().reset_index()
        ctr_by_device['ctr_pct'] = ctr_by_device['ctr'] * 100
        ctr_by_device['ctr_text'] = ctr_by_device['ctr_pct'].apply(lambda val: f"{val:.2f}%")
        ctr_by_device = ctr_by_device.sort_values('ctr_pct', ascending=False)
        
        bars = alt.Chart(ctr_by_device).mark_bar(
            color='#0D9488',
            cornerRadiusTopLeft=6,
            cornerRadiusTopRight=6,
            size=45
        ).encode(
            x=alt.X('ad_device:N', sort='-y', title='Device'),
            y=alt.Y('ctr_pct:Q', title='CTR (%)', scale=alt.Scale(zero=False, padding=20)),
            tooltip=[
                alt.Tooltip('ad_device:N', title='Device'),
                alt.Tooltip('ctr_pct:Q', title='CTR (%)', format='.2f')
            ]
        )

        labels = bars.mark_text(
            align='center',
            baseline='bottom',
            dy=-5,
            fontSize=12,
            fontWeight='bold',
            color='#0F172A'
        ).encode(
            text='ctr_text:N'
        )

        # Average reference line for overall CTR benchmark
        avg_ctr_val = df['ctr'].mean() * 100
        avg_line = alt.Chart(pd.DataFrame({'y': [avg_ctr_val]})).mark_rule(
            color='#EF4444',
            strokeDash=[5, 5],
            strokeWidth=1.5
        ).encode(y='y')

        chart = (bars + labels + avg_line).properties(
            height=300,
            title="Average CTR by Device (vs Overall Benchmark)"
        )
        
        st.altair_chart(chart, use_container_width=True)
        
        # Show best and worst device
        if not ctr_by_device.empty:
            best = ctr_by_device.iloc[0]
            worst = ctr_by_device.iloc[-1]
            st.caption(f"📱 **Best:** {best['ad_device']} ({best['ctr_pct']:.2f}%) | **Worst:** {worst['ad_device']} ({worst['ctr_pct']:.2f}%)")
    
    with col2:
        st.markdown("#### 🕐 CTR Trend (15-Min Intervals)")
        
        # Group by 15-minute intervals (0.25 hour increments)
        df['hour_15min'] = (df['avg_hour'] * 4).round() / 4
        ctr_by_time = df.groupby('hour_15min')['ctr'].mean().reset_index()
        ctr_by_time['ctr_pct'] = ctr_by_time['ctr'] * 100
        ctr_by_time = ctr_by_time.sort_values('hour_15min')
        
        def format_time_label(h_val):
            total_minutes = int(round(h_val * 60 / 15) * 15)
            hours = (total_minutes // 60) % 24
            mins = total_minutes % 60
            period = "AM" if hours < 12 else "PM"
            disp_hour = hours % 12 or 12
            return f"{disp_hour}:{mins:02d} {period}"
            
        ctr_by_time['time_label'] = ctr_by_time['hour_15min'].apply(format_time_label)

        # Smooth curve line chart
        line_chart = alt.Chart(ctr_by_time).mark_line(
            interpolate='monotone',
            color='#0D9488',
            strokeWidth=2.5,
            point=alt.OverlayMarkDef(
                filled=True,
                fill='#0D9488',
                size=40
            )
        ).encode(
            x=alt.X('time_label:N', title='Time Slot (15-Min Granularity)', sort=ctr_by_time['time_label'].tolist()),
            y=alt.Y('ctr_pct:Q', title='CTR (%)', scale=alt.Scale(zero=False, padding=15)),
            tooltip=[
                alt.Tooltip('time_label:N', title='Time Slot'),
                alt.Tooltip('ctr_pct:Q', title='CTR (%)', format='.2f')
            ]
        )

        # Gradient area under curve
        area_chart = alt.Chart(ctr_by_time).mark_area(
            interpolate='monotone',
            opacity=0.15,
            color='#0D9488'
        ).encode(
            x=alt.X('time_label:N', sort=ctr_by_time['time_label'].tolist()),
            y=alt.Y('ctr_pct:Q', scale=alt.Scale(zero=False))
        )

        # Benchmark reference line for overall average CTR
        avg_ctr_val = df['ctr'].mean() * 100
        avg_line = alt.Chart(pd.DataFrame({'y': [avg_ctr_val]})).mark_rule(
            color='#EF4444',
            strokeDash=[5, 5],
            strokeWidth=1.5
        ).encode(y='y')

        chart = (area_chart + line_chart + avg_line).properties(
            height=300,
            title="CTR Performance Trend over 15-Minute Intervals"
        )
        
        st.altair_chart(chart, use_container_width=True)
        
        # Show best and worst time slot
        if not ctr_by_time.empty:
            best = ctr_by_time.loc[ctr_by_time['ctr'].idxmax()]
            worst = ctr_by_time.loc[ctr_by_time['ctr'].idxmin()]
            st.caption(f"⏰ **Peak Slot:** {best['time_label']} ({best['ctr_pct']:.2f}%) | **Lowest Slot:** {worst['time_label']} ({worst['ctr_pct']:.2f}%)")
    
    # Row 2: Platform Performance + Conversion by Category
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### 📊 Platform Performance")
        
        # Determine platform column
        plat_col = None
        if 'platforms_used' in df.columns:
            plat_col = 'platforms_used'
        elif 'platform_source_cleaned' in df.columns:
            plat_col = 'platform_source_cleaned'
            
        if plat_col:
            # Explode the platform column if it contains lists/arrays
            df_exploded = df.copy()
            if df_exploded[plat_col].apply(lambda x: isinstance(x, (list, np.ndarray))).any():
                df_exploded = df_exploded.explode(plat_col)
            
            # Clean platform strings
            df_exploded[plat_col] = df_exploded[plat_col].fillna('unknown').astype(str).str.lower().str.strip()
            df_exploded = df_exploded[df_exploded[plat_col] != '']
            
            # Calculate average ROAS per platform
            roas_col = 'roas' if 'roas' in df_exploded.columns else 'platform_avg_roas'
            platform_perf = df_exploded.groupby(plat_col)[roas_col].mean().reset_index()
            platform_perf = platform_perf.sort_values(roas_col, ascending=False)
            
            chart = alt.Chart(platform_perf).mark_bar(
                color='#0D9488',
                cornerRadiusTopLeft=5,
                cornerRadiusTopRight=5
            ).encode(
                x=alt.X(f'{plat_col}:N', sort='-y', title='Platform'),
                y=alt.Y(f'{roas_col}:Q', title='Avg ROAS (x)'),
                tooltip=[
                    alt.Tooltip(f'{plat_col}:N', title='Platform'),
                    alt.Tooltip(f'{roas_col}:Q', title='ROAS', format='.2f')
                ]
            ).properties(
                height=280,
                title="ROAS by Platform"
            )
            
            st.altair_chart(chart, use_container_width=True)
            
            if not platform_perf.empty:
                best = platform_perf.iloc[0]
                st.caption(f"🏆 **Best Platform:** {best[plat_col].upper()} ({best[roas_col]:.2f}x)")
        else:
            st.warning("Platform data not available in current dataset")
    
    with col4:
        st.markdown("#### 🔄 Conversion Rate by Category")
        
        conv_by_category = df.groupby('ad_category')['conversion_rate'].mean().reset_index()
        conv_by_category['conv_pct'] = conv_by_category['conversion_rate'] * 100
        conv_by_category = conv_by_category.sort_values('conv_pct', ascending=False)
        
        chart = alt.Chart(conv_by_category).mark_bar(
            color='#0D9488',
            cornerRadiusTopLeft=5,
            cornerRadiusTopRight=5
        ).encode(
            x=alt.X('ad_category:N', sort='-y', title='Category'),
            y=alt.Y('conv_pct:Q', title='Conversion Rate (%)'),
            tooltip=[
                alt.Tooltip('ad_category:N', title='Category'),
                alt.Tooltip('conv_pct:Q', title='Conversion Rate', format='.2f')
            ]
        ).properties(
            height=280,
            title="Conversion Rate by Ad Category"
        )
        
        st.altair_chart(chart, use_container_width=True)
        
        if not conv_by_category.empty:
            best = conv_by_category.iloc[0]
            st.caption(f"🎯 **Highest Conversion:** {best['ad_category']} ({best['conv_pct']:.2f}%)")
    
    # Row 3: Quick Actions
    st.markdown("### ⚡ Quick Optimization Actions")
    st.caption("Actionable recommendations based on current performance data")
    
    actions = []
    
    # Best device
    if not ctr_by_device.empty:
        best_device = ctr_by_device.iloc[0]
        actions.append(f"📱 **Increase Budget for {best_device['ad_device']} Ads** (CTR: {best_device['ctr_pct']:.1f}% - Highest customer click rate)")
    
    # Worst device
    if len(ctr_by_device) > 1:
        worst_device = ctr_by_device.iloc[-1]
        actions.append(f"📉 **Reduce Spend on {worst_device['ad_device']} Ads** (CTR: {worst_device['ctr_pct']:.1f}% - Lowest customer click rate)")
    
    # Best time slot
    if not ctr_by_time.empty:
        best_time = ctr_by_time.loc[ctr_by_time['ctr'].idxmax()]
        h_str = best_time['time_label']
        actions.append(f"⏰ **Run Ads Around {h_str}** (CTR: {best_time['ctr_pct']:.1f}% - Most active audience time)")
    
    # Best platform
    if 'platform_source_cleaned' in df.columns and not platform_perf.empty:
        best_platform = platform_perf.iloc[0]
        actions.append(f"📊 **Increase Budget for {best_platform['platform_source_cleaned'].upper()}** (ROAS: {best_platform['platform_avg_roas']:.2f}x - Top return on spend)")
    
    # Highest conversion category
    if not conv_by_category.empty:
        best_conv = conv_by_category.iloc[0]
        actions.append(f"🔄 **Invest More in {best_conv['ad_category']} Category** (Conversion Rate: {best_conv['conv_pct']:.1f}% - Highest sales conversions)")
    
    # Display actions in columns
    cols = st.columns(2)
    for i, action in enumerate(actions[:4]):  # Limit to 4 actions
        with cols[i % 2]:
            if "Increase" in action or "Invest More" in action:
                st.success(action)
            elif "Reduce" in action:
                st.error(action)
            elif "Run Ads" in action:
                st.warning(action)
            else:
                st.info(action)
    
    # If no actions, show message
    if not actions:
        st.info("📊 No optimization actions available. Add more data to get insights.")
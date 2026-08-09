"""
Page 3: Performance Insights
Advanced Pandas-based business analytics and role-based operational insights
"""

import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

def show(df_gold, user_type):
    """Display Performance Insights page"""
    
    st.markdown("🎯 *Analytics Console — 💡 **How it works:** Explore underlying data trends across categories, channels, and target cohorts.*")
    
    if df_gold.empty:
        st.warning("⚠️ No data available to compile insights.")
        return
        
    df = df_gold.copy()
    
    # Standardize types and clean numeric columns
    numeric_cols = [
        'total_ad_spend', 'total_revenue', 'cost_per_click', 'avg_hour', 
        'avg_user_age', 'roas', 'ctr', 'conversion_rate', 'engagement_score', 
        'cost_efficiency_score', 'avg_watch_ratio', 'avg_watch_duration', 
        'avg_ded_score', 'category_age_affinity'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
    # Age binning function helper
    def get_age_group(age):
        if age <= 25:
            return "18-25 (Gen Z)"
        elif age <= 35:
            return "26-35 (Millennials)"
        elif age <= 50:
            return "36-50 (Gen X)"
        else:
            return "50+ (Boomers)"
            
    df['age_group'] = df['avg_user_age'].apply(get_age_group)
    
    # Calculate Engagement Efficiency derived metric (Item 11)
    # Handle division by zero
    df['engagement_efficiency'] = np.where(df['cost_per_click'] > 0, df['engagement_score'] / df['cost_per_click'], 0.0)

    # Tabs for logical layout
    tab1, tab2, tab3 = st.tabs(["🎨 Creatives & Platforms", "👥 Demographic Profiling", "⚙️ Operational Efficiency"])
    
    # ============================================================
    # TAB 1: CREATIVES & PLATFORMS
    # ============================================================
    with tab1:
        st.markdown("### 🎨 Creative & Platform Performance Analysis")
        
        # Insight 5: Ad Type Performance
        st.markdown("#### Ad Type Performance")
        st.write("Average Click-Through Rate (CTR) and ROAS across ad formats:")
        
        ad_type_stats = df.groupby('ad_type').agg({
            'ctr': 'mean',
            'roas': 'mean',
            'Ad_Reference_ID': 'count'
        }).reset_index().rename(columns={'Ad_Reference_ID': 'Volume'})
        ad_type_stats['ctr_pct'] = ad_type_stats['ctr'] * 100
        
        # Color based on profitability threshold
        ad_type_stats['color'] = ad_type_stats['roas'].apply(
            lambda x: '#22C55E' if x >= 2.0 else '#EF4444'
        )
        
        # Pre-format columns and rename headers for polished display
        ad_type_display = ad_type_stats.copy()
        ad_type_display['ctr_pct'] = ad_type_display['ctr_pct'].apply(lambda x: f"{x:.2f}%")
        ad_type_display['roas'] = ad_type_display['roas'].apply(lambda x: f"{x:.2f}x")
        ad_type_display = ad_type_display.rename(columns={
            'ad_type': 'Ad Type',
            'roas': 'Average ROAS',
            'ctr_pct': 'CTR (%)'
        })
        
        # Display the data table
        ad_type_display['Volume'] = ad_type_display['Volume'].astype(str)
        st.dataframe(ad_type_display[['Ad Type', 'CTR (%)', 'Average ROAS', 'Volume']], hide_index=True, use_container_width=True)
        
        # Display the chart below it with full container width
        chart = alt.Chart(ad_type_stats).mark_bar(
            cornerRadiusTopLeft=5,
            cornerRadiusTopRight=5
        ).encode(
            x=alt.X('ad_type:N', title='Ad Type', sort=['Carousel', 'Image', 'Text', 'Video'], axis=alt.Axis(labelAngle=0)),
            y=alt.Y('roas:Q', title='Average ROAS (x)'),
            color=alt.Color('color:N', scale=None, legend=None),
            tooltip=['ad_type', 'Volume', 'roas']
        ).properties(height=250, title="ROAS by Ad Type")
        st.altair_chart(chart, use_container_width=True)
        st.caption("🟢 Green = Profitable (≥2.0x) | 🔴 Red = Below Target")
            
        st.markdown("---")
        st.markdown("#### Platform Performance")
        st.write("Distribution of ad spend, revenue, and return metrics across delivery networks:")
        
        # Determine platform column
        plat_col = None
        if 'platforms_used' in df.columns:
            plat_col = 'platforms_used'
        elif 'platform_source_cleaned' in df.columns:
            plat_col = 'platform_source_cleaned'
            
        if plat_col:
            df_exploded = df.copy()
            # If the platform column contains list/array objects, extract the primary platform
            if df_exploded[plat_col].apply(lambda x: isinstance(x, (list, np.ndarray))).any():
                df_exploded[plat_col] = df_exploded[plat_col].apply(
                    lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else x
                )
            
            # Clean platform names and group unknown/none values into 'Other'
            df_exploded[plat_col] = df_exploded[plat_col].fillna('other').astype(str).str.lower().str.strip()
            df_exploded[plat_col] = df_exploded[plat_col].replace({
                'unknown': 'other',
                'none': 'other',
                '': 'other'
            })
            df_exploded = df_exploded[df_exploded[plat_col] != '']
            
            platform_stats = df_exploded.groupby(plat_col).agg({
                'total_ad_spend': 'sum',
                'total_revenue': 'sum',
                'roas': 'mean'
            }).reset_index()
            
            # Clean up display columns
            platform_stats = platform_stats.rename(columns={plat_col: 'Platform'})
            # Title case for platform names (Facebook, Google, Instagram, Other)
            platform_stats['Platform'] = platform_stats['Platform'].str.title()

            # Format and rename columns for clean display — convert to str so all left-align
            platform_display = platform_stats.copy()
            platform_display['total_ad_spend'] = platform_display['total_ad_spend'].apply(lambda x: f"${x:,.2f}")
            platform_display['total_revenue']  = platform_display['total_revenue'].apply(lambda x: f"${x:,.2f}")
            platform_display['roas']           = platform_display['roas'].apply(lambda x: f"{x:.2f}x")
            platform_display = platform_display.rename(columns={
                'total_ad_spend': 'Ad Spend',
                'total_revenue':  'Total Revenue',
                'roas':           'ROAS'
            })

            # Display the platform data table
            st.dataframe(
                platform_display[['Platform', 'Ad Spend', 'Total Revenue', 'ROAS']],
                use_container_width=True,
                hide_index=True
            )
            
            # Display the platform chart below it
            st.markdown("##### Revenue Generation by Platform")
            chart = alt.Chart(platform_stats).mark_bar(
                color='#22C55E',
                cornerRadiusTopLeft=5,
                cornerRadiusTopRight=5
            ).encode(
                x=alt.X('Platform:N', title='Platform', sort=['FACEBOOK', 'GOOGLE', 'INSTAGRAM', 'OTHER'], axis=alt.Axis(labelAngle=0)),
                y=alt.Y('total_revenue:Q', title='Revenue Generated ($)', axis=alt.Axis(format='$,.0f')),
                tooltip=['Platform', alt.Tooltip('total_revenue', title='Revenue', format='$,.2f'), alt.Tooltip('total_ad_spend', title='Spend', format='$,.2f')]
            ).properties(height=250)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("ℹ️ Platform column not found in active dataset.")

    # ============================================================
    # TAB 2: DEMOGRAPHIC PROFILING
    # ============================================================
    with tab2:
        st.markdown("### 👥 Demographic & Age Analysis")
        
        # Insight 6: Age Group Usage by Category
        st.markdown("#### Age Group Usage by Ad Category")
        st.write("Volume of ads targeting specific demographic segments by product category:")
        
        age_cat_usage = df.groupby(['ad_category', 'age_group']).size().reset_index(name='Ad Count')
        # Clean age label to remove (Gen Z), (Millennials), etc.
        age_cat_usage['Age Bracket'] = age_cat_usage['age_group'].str.extract(r'^(\S+)')
        
        # Calculate percentage share for each category
        cat_totals = age_cat_usage.groupby('ad_category')['Ad Count'].transform('sum')
        age_cat_usage['Share (%)'] = (age_cat_usage['Ad Count'] / cat_totals * 100).round(2)
        
        # Check if we are viewing a single category or all categories
        is_single_category = df['ad_category'].nunique() <= 1
        
        if is_single_category:
            category_name = df['ad_category'].iloc[0] if not df.empty else "Selected Category"
            st.markdown(f"##### Demographic Target Share - {category_name}")

            import plotly.express as px

            fig = px.pie(
                age_cat_usage,
                values='Ad Count',
                names='Age Bracket',
                hole=0.5,
                color='Age Bracket',
                color_discrete_map={
                    '18-25': '#4E79A7',
                    '26-35': '#F28E2B',
                    '36-50': '#E15759',
                    '50+': '#76B7B2'
                }
            )

            fig.update_traces(
                textinfo='percent',
                textposition='inside',
                insidetextfont=dict(size=14, color='white', family='Arial Black'),
                hovertemplate="<b>Age Group:</b> %{label}<br><b>Count:</b> %{value}<br><b>Share:</b> %{percent}<extra></extra>"
            )

            fig.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                height=280,
                legend=dict(
                    title_text="",
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.05
                )
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            chart = alt.Chart(age_cat_usage).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                x=alt.X('ad_category:N', title='Category', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('Share (%):Q', title='Demographic Share (%)'),
                color=alt.Color('Age Bracket:N', legend=alt.Legend(title=None), scale=alt.Scale(scheme='tableau10')),
                xOffset='Age Bracket:N',
                tooltip=[
                    alt.Tooltip('ad_category:N', title='Category'),
                    alt.Tooltip('Age Bracket:N', title='Age Bracket'),
                    alt.Tooltip('Ad Count:Q', title='Ad Count'),
                    alt.Tooltip('Share (%):Q', title='Share (%)', format='.2f')
                ]
            ).properties(
                height=250
            ).configure_axis(
                titlePadding=10
            )
            st.altair_chart(chart, use_container_width=True)
        
        st.markdown("---")
        
        # Insight 9: Category Recommendations by Age (Top Category by ROAS per age bracket)
        st.markdown("#### Category Recommendations by Age Group")
        st.write("Best performing product categories (by ROAS) for each age cohort:")
        
        age_cat_roas = df.groupby(['age_group', 'ad_category'])['roas'].mean().reset_index()
        # Find index of max ROAS for each age group
        idx = age_cat_roas.groupby('age_group')['roas'].idxmax()
        top_cats_by_age = age_cat_roas.loc[idx].sort_values('roas', ascending=False).copy()
        
        # Clean age_group string to remove (Gen Z), (Millennials), etc.
        top_cats_by_age['age_group'] = top_cats_by_age['age_group'].str.extract(r'^(\S+)')
        
        # Pre-format columns and rename headers
        top_cats_by_age['roas'] = top_cats_roas_val if 'top_cats_roas_val' in locals() else top_cats_by_age['roas'].apply(lambda x: f"{x:.2f}x")
        top_cats_by_age = top_cats_by_age.rename(columns={
            'age_group': 'Age Group',
            'ad_category': 'Top Category',
            'roas': 'Average ROAS'
        })
        st.dataframe(top_cats_by_age, hide_index=True, use_container_width=True)
        
        st.markdown("---")
        
        # Insight 10: Optimal Time by Demographic
        st.markdown("#### Optimal Time by Demographic Bracket")
        st.write("Best performing hour of day (highest average CTR) for each age group:")
        
        # Round avg_hour to nearest whole hour for cleaner grouping
        df['hour_binned'] = df['avg_hour'].round().astype(int)
        dem_time_stats = df.groupby(['age_group', 'hour_binned'])['ctr'].mean().reset_index()
        
        idx_time = dem_time_stats.groupby('age_group')['ctr'].idxmax()
        opt_time_by_dem = dem_time_stats.loc[idx_time].copy()
        
        # Clean age_group string to remove (Gen Z), (Millennials), etc.
        opt_time_by_dem['age_group'] = opt_time_by_dem['age_group'].str.extract(r'^(\S+)')
        
        # Pre-format and rename columns
        opt_time_by_dem['ctr_pct'] = (opt_time_by_dem['ctr'] * 100).apply(lambda x: f"{x:.3f}%")
        
        def to_ampm(h):
            h = int(h)
            if h == 0:
                return "12:00 AM"
            elif h < 12:
                return f"{h}:00 AM"
            elif h == 12:
                return "12:00 PM"
            else:
                return f"{h-12}:00 PM"
                
        opt_time_by_dem['hour_binned'] = opt_time_by_dem['hour_binned'].apply(to_ampm)
        opt_time_by_dem = opt_time_by_dem.rename(columns={
            'age_group': 'Age Group',
            'hour_binned': 'Peak Hour',
            'ctr_pct': 'Click-Through Rate'
        })
        st.dataframe(opt_time_by_dem[['Age Group', 'Peak Hour', 'Click-Through Rate']], hide_index=True, use_container_width=True)
        



    # ============================================================
    # TAB 3: OPERATIONAL EFFICIENCY
    # ============================================================
    with tab3:
        st.markdown("### ⚙️ Operational Efficiency")

        # ── Section 1: Device × Format ROAS Matrix ──────────────────────
        st.markdown("#### 📱 Device × Ad Format Performance Matrix")
        st.write("Which device + format combination delivers the best ROAS? (Higher = better return per dollar spent)")

        if 'ad_device' in df.columns and 'ad_type' in df.columns and 'roas' in df.columns:
            # Build pivot: rows = Ad Format, cols = Device
            matrix_df = df.groupby(['ad_type', 'ad_device'])['roas'].mean().reset_index()

            # Filter out All-Devices for cleaner matrix
            matrix_df = matrix_df[~matrix_df['ad_device'].str.lower().str.contains('all', na=False)]

            if not matrix_df.empty:
                pivot = matrix_df.pivot(index='ad_type', columns='ad_device', values='roas').round(2)

                # Find best combo
                best_combo_val = matrix_df['roas'].max()
                best_row = matrix_df.loc[matrix_df['roas'].idxmax()]
                best_fmt_combo = best_row['ad_type']
                best_dev_combo = best_row['ad_device']

                # Render heatmap-style HTML table
                devices = pivot.columns.tolist()
                formats = pivot.index.tolist()

                # Build header
                header_cols = "".join([f"<th style='padding:10px 18px;text-align:center;font-size:13px;font-weight:700;color:#475569;background:#F1F5F9;'>{d}</th>" for d in devices])
                rows_html = ""
                for fmt in formats:
                    cells = ""
                    for dev in devices:
                        val = pivot.loc[fmt, dev] if dev in pivot.columns and fmt in pivot.index else None
                        if val is None or pd.isna(val):
                            cells += "<td style='padding:10px 18px;text-align:center;color:#CBD5E1;'>—</td>"
                        else:
                            is_best = (fmt == best_fmt_combo and dev == best_dev_combo)
                            if is_best:
                                bg = "#DCFCE7"; color = "#15803D"; fw = "800"; border = "2px solid #22C55E"
                            elif val >= 2.75:
                                bg = "#F0FDF4"; color = "#16A34A"; fw = "700"; border = "1px solid #BBF7D0"
                            elif val >= 2.0:
                                bg = "#FFFBEB"; color = "#B45309"; fw = "600"; border = "1px solid #FDE68A"
                            else:
                                bg = "#FEF2F2"; color = "#DC2626"; fw = "600"; border = "1px solid #FECACA"
                            star = " 🏆" if is_best else ""
                            cells += f"<td style='padding:10px 18px;text-align:center;background:{bg};border:{border};border-radius:6px;font-size:14px;font-weight:{fw};color:{color};'>{val:.2f}x{star}</td>"
                    rows_html += f"<tr><td style='padding:10px 18px;font-weight:700;font-size:13px;color:#1e293b;background:#F8FAFC;'>{fmt}</td>{cells}</tr>"

                table_html = f"""
                <div style="overflow-x:auto;margin-bottom:8px;">
                <table style="width:100%;border-collapse:separate;border-spacing:4px;font-family:'Segoe UI',sans-serif;">
                    <thead>
                        <tr>
                            <th style="padding:10px 18px;text-align:left;font-size:13px;font-weight:700;color:#475569;background:#F1F5F9;">Ad Format</th>
                            {header_cols}
                        </tr>
                    </thead>
                    <tbody>{rows_html}</tbody>
                </table>
                </div>
                """
                st.markdown(table_html, unsafe_allow_html=True)
                st.caption("🏆 Best combo | 🟢 High ROAS (≥2.75x) | 🟡 Moderate (≥2.0x) | 🔴 Below Target")

                # Link to ML Forecaster
                st.info(f"💡 **Best Combo: {best_fmt_combo} on {best_dev_combo}** ({best_combo_val:.2f}x ROAS) — Use this combination in the **ML Performance Forecaster** to predict your next campaign's results.")
            else:
                st.info("ℹ️ Not enough device breakdown data to build the matrix.")
        else:
            st.info("ℹ️ Device, format or ROAS columns not found in active dataset.")

        st.markdown("---")

        # ── Section 2: Best Day to Run Ads ──────────────────────────────
        st.markdown("#### 🗓️ Best Day to Run Ads")
        st.write("Average CTR by day of the week — run campaigns on peak days for maximum clicks:")

        day_names = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
        df['day_name'] = df['best_day'].map(day_names)

        weekly_stats = df.groupby(['best_day', 'day_name']).agg(
            ctr=('ctr', 'mean'),
            roas=('roas', 'mean')
        ).reset_index().sort_values('best_day')

        weekly_stats['ctr_pct'] = (weekly_stats['ctr'] * 100).round(2)
        weekly_stats['roas']    = weekly_stats['roas'].round(2)

        best_day = weekly_stats.loc[weekly_stats['ctr_pct'].idxmax(), 'day_name'] if not weekly_stats.empty else "—"

        chart = alt.Chart(weekly_stats).mark_line(
            point=alt.OverlayMarkDef(filled=True, fill='#0D9488', size=60),
            color='#0D9488',
            strokeWidth=2.5
        ).encode(
            x=alt.X('day_name:N',
                    sort=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                    title='Day of Week',
                    axis=alt.Axis(labelAngle=0)),
            y=alt.Y('ctr_pct:Q', title='Average CTR (%)'),
            tooltip=['day_name', alt.Tooltip('ctr_pct', title='CTR (%)', format='.2f'), alt.Tooltip('roas', title='ROAS', format='.2f')]
        ).properties(height=230)
        st.altair_chart(chart, use_container_width=True)
        st.caption(f"📅 Peak CTR day: **{best_day}** — Schedule high-priority campaigns on this day for best engagement.")


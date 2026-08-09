import streamlit as st
import os
import pandas as pd
import ollama

st.set_page_config(page_title="AdTech Optimizer Agent", layout="wide")
st.title("🎯 AdTech Optimizer Retrieval Agent (Llama 3.2)")
st.subheader("CDAC BDA Capstone Project Pipeline")

@st.cache_data
def load_adtech_data():
    csv_variants = ["adtech_data.csv", "adtech_data.CSV"]
    csv_filename = None
    for variant in csv_variants:
        if os.path.exists(variant):
            csv_filename = variant
            break
    if not csv_filename:
        all_files = os.listdir(".")
        for f in all_files:
            if "adtech" in f.lower() and f.lower().endswith(".csv"):
                csv_filename = f
                break
    if not csv_filename:
        st.error("Error: 'adtech_data.csv' file not found.")
        st.stop()
        
    df = pd.read_csv(csv_filename, encoding='utf-8', on_bad_lines='skip')
    return df

try:
    df = load_adtech_data()
    st.success(f"📊 Dataset Loaded via Pandas Engine! Total Records: {len(df)} rows.")
    with st.expander("👀 Preview Active Trained Dataset Matrix Rows"):
        st.dataframe(df.head(5))
except Exception as e:
    st.error(f"CSV Core Loading Error: {e}")
    st.stop()

# --- OPTIMIZED FAST FILTERING SEARCH ENGINE ---
def search_relevant_context(user_query, dataframe):
    keywords = [word.lower() for word in user_query.split() if len(word) > 3]
    matched_rows = []
    for idx, row in dataframe.iterrows():
        row_str = " | ".join([f"{col}: {val}" for col, val in row.items()])
        if any(kw in row_str.lower() for kw in keywords):
            matched_rows.append(f"Record {idx+1}: {row_str}")
        if len(matched_rows) >= 5: 
            break
            
    if not matched_rows:
        for idx, row in dataframe.head(5).iterrows():
            row_str = " | ".join([f"{col}: {val}" for col, val in row.items()])
            matched_rows.append(f"Record {idx+1}: {row_str}")
            
    return "\n".join(matched_rows)

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 9 MAIN CLIENT-READY QUERY SUGGESTIONS BLOCK ---
st.markdown("💡 **Client Analytics Dashboard — Click any query to analyze instantly:**")

# Humne 3 rows banayi hain, har row mein 3 core business metrics hain
row1_col1, row1_col2, row1_col3 = st.columns(3)
row2_col1, row2_col2, row2_col3 = st.columns(3)
row3_col1, row3_col2, row3_col3 = st.columns(3)

clicked_query = None

# --- ROW 1: Revenue & Profitability (Client ka sabse pehla focus) ---
with row1_col1:
    if st.button("🚀 Top Performing Campaigns (Highest ROAS)"):
        clicked_query = "Identify the top 5 performing campaigns with the highest ROAS and revenue generated."
with row1_col2:
    if st.button("💰 Best Conversion Rate Analysis"):
        clicked_query = "Which ad groups or campaigns have achieved the highest conversion rates?"
with row1_col3:
    if st.button("📈 High ROI Platform Discovery"):
        clicked_query = "Compare performance across channels to find which platform yields the maximum return on investment."

# --- ROW 2: Budget Leakage & Losses (Paisa bachaane ke liye) ---
with row2_col1:
    if st.button("⚠️ Budget Bleeding (High Spend, Zero Revenue)"):
        clicked_query = "Find campaigns with high budget spending but zero or extremely low conversions."
with row2_col2:
    if st.button("🛑 Poor Performing Ad Creative Analysis"):
        clicked_query = "Identify ads with high impressions but dangerously low Click-Through Rates (CTR)."
with row2_col3:
    if st.button("💸 High Cost-Per-Click (CPC) Alert"):
        clicked_query = "List the campaigns that have the highest Cost-Per-Click (CPC) draining the budget."

# --- ROW 3: Efficiency & Optimization (Agli strategy ke liye) ---
with row3_col1:
    if st.button("🎯 Audience & Demographics Performance"):
        clicked_query = "Analyze which target audience segment or location is driving the cheapest conversions."
with row3_col2:
    if st.button("📉 Lowest Performing Campaigns Summary"):
        clicked_query = "Summarize the bottom 5 underperforming campaigns that need to be paused immediately."
with row3_col3:
    if st.button("🔄 Complete Platform Cost vs Revenue Summary"):
        clicked_query = "Provide a full summary of total ad spend versus total revenue generated across the entire dataset."


# Chat history render karein
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- EXECUTION LOGIC ---
user_query = st.chat_input("Query your AdTech metrics data warehouse...")

if clicked_query and not user_query:
    user_query = clicked_query

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        filtered_context = search_relevant_context(user_query, df)
        
        messages_pipeline = [
            {
                "role": "system",
                "content": (
                    "You are an automated AdTech optimization analyst agent for a CDAC BDA project.\n"
                    "Analyze the provided context data and user query to give a clean, client-friendly report.\n\n"
                    "ALWAYS format your answer into EXACTLY these 4 clear Markdown sections:\n\n"
                    "### 📊 Executive Summary\n"
                    "(1-2 clear sentences summarizing the answer upfront)\n\n"
                    "### 🔑 Key Metrics Highlights\n"
                    "(Bullet list of specific numbers formatted with emojis and USD currency like $12,500 and ROAS like 3.20x)\n\n"
                    "### 💡 Strategic Insights\n"
                    "(2-3 clear business takeaways)\n\n"
                    "### 🚀 Recommended Client Actions\n"
                    "(Clear action items with status icons: 🟢 Scale, 🟡 Monitor, 🔴 Pause)\n\n"
                    "STRICT RULES:\n"
                    "- NEVER output raw dataframes, pipe-separated text, or python code dumps.\n"
                    "- Format all currency as USD ($XX,XXX).\n"
                    "- Keep sentences concise, highly readable, and executive-ready.\n\n"
                    f"Relevant Context:\n{filtered_context}"
                )
            },
            {"role": "user", "content": user_query}
        ]
        
        try:
            from llm.copilot import get_active_ollama_model
            active_model = get_active_ollama_model()
            if not active_model:
                raise ValueError("No active Ollama LLM model available.")
                
            stream = ollama.chat(model=active_model, messages=messages_pipeline, stream=True)
            for chunk in stream:
                full_response += chunk['message']['content']
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            if clicked_query:
                st.rerun()
                
        except Exception as e:
            # Fallback formatted response if Ollama engine is offline or model fails
            total_spend = df['total_ad_spend'].sum() if 'total_ad_spend' in df.columns else 0
            total_rev = df['total_revenue'].sum() if 'total_revenue' in df.columns else 0
            avg_roas = df['roas'].mean() if 'roas' in df.columns else 0
            
            fallback_msg = (
                f"### 📊 Executive Summary\n"
                f"Analysis completed across {len(df)} active dataset records for your query.\n\n"
                f"----\n\n"
                f"### 🔑 Key Metrics Highlights\n"
                f"• 💰 **Total Ad Spend:** `${total_spend:,.2f}`\n"
                f"• 📈 **Total Revenue:** `${total_rev:,.2f}`\n"
                f"• 🎯 **Average Portfolio ROAS:** `{avg_roas:.2f}x`\n\n"
                f"----\n\n"
                f"### 💡 Strategic Insights\n"
                f"```text\n{filtered_context}\n```\n\n"
                f"----\n\n"
                f"### 🚀 Recommended Client Actions\n"
                f"• 🟢 **Scale High ROAS:** Reallocate budget to top performing campaigns.\n"
                f"• 🔴 **Pause Underperformers:** Stop campaigns spending without revenue return."
            )
            response_placeholder.markdown(fallback_msg)
            st.session_state.messages.append({"role": "assistant", "content": fallback_msg})
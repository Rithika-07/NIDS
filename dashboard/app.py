import os
import redis
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

load_dotenv()

# Setup config
REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))

st.set_page_config(
    page_title="Production NIDS Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# Auto-refresh every 2 seconds
count = st_autorefresh(interval=2000, limit=None, key="nids_autorefresh")

@st.cache_resource
def get_redis_connection():
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

try:
    r = get_redis_connection()
    r.ping()
except redis.ConnectionError:
    st.error(f"Cannot connect to Redis at {REDIS_HOST}:{REDIS_PORT}. Is it running?")
    st.stop()

# Title & Description
st.title("🛡️ Network Intrusion Detection System (NIDS)")
st.markdown("""
This dashboard monitors live network traffic via Redis Streams. 
It performs binary classification and multi-class classification, with **Gemini AI Explainability** for anomalies.
""")

# Fetch Data from Redis
# We use XREVRANGE to get the most recent alerts (last 1000 for stats, last 100 for display)
try:
    raw_alerts = r.xrevrange('nids:alerts', max='+', min='-', count=1000)
    explanations = r.hgetall('nids:explanations')
except Exception as e:
    st.error(f"Error reading from Redis: {e}")
    raw_alerts = []
    explanations = {}

if not raw_alerts:
    st.info("Waiting for network traffic... (Ensure producer and worker are running)")
    st.stop()

# Parse alerts
parsed_alerts = []
for msg_id, data in raw_alerts:
    parsed_alerts.append(data)

df_alerts = pd.DataFrame(parsed_alerts)
df_alerts['is_anomaly'] = df_alerts['is_anomaly'].map({'true': True, 'false': False})

# Calculate Metrics
total_flows = r.xlen('nids:alerts')
recent_anomalies = df_alerts['is_anomaly'].sum()
recent_normal = len(df_alerts) - recent_anomalies
anomaly_rate = (recent_anomalies / len(df_alerts)) * 100 if len(df_alerts) > 0 else 0

# Dashboard Layout
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Flows Analyzed", f"{total_flows:,}")
col2.metric("Recent Normal (last 1000)", f"{recent_normal:,}")
col3.metric("Recent Anomalies", f"{recent_anomalies:,}", delta=f"{anomaly_rate:.1f}% Rate", delta_color="inverse")
col4.metric("AI Explanations Available", f"{len(explanations)}")

st.divider()

# Main Feed & Sidebar
col_feed, col_sidebar = st.columns([2, 1])

with col_feed:
    st.subheader("📡 Live Alert Feed")
    
    # Display the last 50 alerts
    display_df = df_alerts.head(50)
    
    for _, row in display_df.iterrows():
        is_anomaly = row.get('is_anomaly', False)
        timestamp = row.get('timestamp', 'Unknown Time')
        attack_cat = row.get('attack_category', 'Unknown')
        src_bytes = row.get('sbytes', 0)
        proto = row.get('proto', '')
        
        if is_anomaly and attack_cat != 'Normal':
            with st.container(border=True):
                st.markdown(f"**🚨 {attack_cat} Attack** | `{timestamp}`")
                st.markdown(f"Protocol: `{proto}` | Source Bytes: `{src_bytes}`")
                
                # Fetch explanation from Hash
                explanation = explanations.get(attack_cat, "Explanation loading... (Waiting for Gemini batch)")
                
                with st.expander("▼ AI Explanation (Hash)", expanded=True):
                    st.info(explanation)
        else:
            with st.container(border=True):
                st.markdown(f"**🟢 Normal** | `{timestamp}`")
                st.markdown(f"Protocol: `{proto}` | Source Bytes: `{src_bytes}`")

with col_sidebar:
    st.subheader("📊 Threat Distribution (Recent)")
    
    anomaly_df = df_alerts[df_alerts['is_anomaly'] == True]
    if not anomaly_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        attack_counts = anomaly_df['attack_category'].value_counts()
        sns.barplot(x=attack_counts.values, y=attack_counts.index, palette="rocket", ax=ax, orient='h')
        ax.set_xlabel("Count")
        ax.set_ylabel("Attack Category")
        st.pyplot(fig)
    else:
        st.write("No anomalies detected recently.")
        
    st.divider()
    
    st.subheader("📈 Model Metrics")
    st.markdown("Pre-generated during training.")
    
    if os.path.exists("roc_pr_curves.png"):
        st.image("roc_pr_curves.png", caption="Model ROC and Precision-Recall Curves", use_column_width=True)
    
    if os.path.exists("shap_summary.png"):
        st.image("shap_summary.png", caption="SHAP Feature Importance", use_column_width=True)

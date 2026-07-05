import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Production NIDS Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# Title & Description
st.title("🛡️ Network Intrusion Detection System (NIDS)")
st.markdown("""
This dashboard monitors network traffic using a production-grade machine learning pipeline. 
It performs binary classification (Normal vs Anomaly) and multi-class classification (Attack Category) with predictive fallback overrides.
""")

# Load Models
@st.cache_resource
def load_pipelines():
    bin_path = 'nids_bin_pipeline.pkl'
    multi_path = 'nids_multi_pipeline.pkl'
    if not os.path.exists(bin_path) or not os.path.exists(multi_path):
        return None, None
    bin_data = joblib.load(bin_path)
    multi_data = joblib.load(multi_path)
    return bin_data, multi_data

bin_data, multi_data = load_pipelines()

if bin_data is None or multi_data is None:
    st.error("⚠️ Serialized models not found! Please run `python src/train.py` to train and save the production pipelines.")
else:
    bin_pipeline = bin_data['pipeline']
    multi_pipeline = multi_data['pipeline']
    selected_features = bin_data['features']
    
    st.success("✅ Models loaded successfully!")
    
    # Sidebar features list
    st.sidebar.title("Pipeline Settings")
    st.sidebar.markdown(f"**Number of Selected Features:** {len(selected_features)}")
    with st.sidebar.expander("Show Features"):
        st.write(selected_features)

    # 1. Main CSV Uploader
    st.header("🔍 Analyze Traffic (CSV)")
    uploaded_file = st.file_uploader("Upload network traffic CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            st.write(f"Loaded {df_input.shape[0]} rows and {df_input.shape[1]} columns.")
            
            # Check for missing features
            missing_features = [f for f in selected_features if f not in df_input.columns]
            if missing_features:
                st.error(f"❌ The uploaded CSV is missing the following required features: {missing_features}")
            else:
                # Perform predictions
                with st.spinner("Processing traffic through ML pipelines..."):
                    # 1. Binary Predictions
                    bin_preds = bin_pipeline.predict(df_input)
                    
                    # 2. Multi-class Predictions
                    multi_preds = multi_pipeline.predict(df_input)
                    
                    # 3. Apply Prediction Fallback Logic (Unknown handling)
                    final_preds = []
                    fallbacks_count = 0
                    for bp, mp in zip(bin_preds, multi_preds):
                        if bp == 1 and mp == 'Normal':
                            final_preds.append('Unknown')
                            fallbacks_count += 1
                        else:
                            final_preds.append(mp)
                            
                st.info(f"⚡ Inconsistent predictions handled: {fallbacks_count} anomalies reclassified from 'Normal' to 'Unknown'.")
                
                # Combine results
                df_results = df_input.copy()
                df_results['Prediction'] = bin_preds
                df_results['Prediction_Label'] = ['Anomaly' if p == 1 else 'Normal' for p in bin_preds]
                df_results['Attack_Category'] = final_preds
                
                # Metrics Display
                col1, col2, col3, col4 = st.columns(4)
                total_packets = len(df_results)
                anomalies = int(df_results['Prediction'].sum())
                normal = total_packets - anomalies
                anomaly_rate = (anomalies / total_packets) * 100
                
                col1.metric("Total Packets Analyzed", f"{total_packets:,}")
                col2.metric("Normal Traffic", f"{normal:,}")
                col3.metric("Anomalies Detected", f"{anomalies:,}", delta=f"{anomaly_rate:.2f}% Rate", delta_color="inverse")
                col4.metric("Inconsistencies Overridden", f"{fallbacks_count:,}")

                # Threat Distribution Chart
                st.subheader("📊 Threat Distribution")
                fig, ax = plt.subplots(figsize=(10, 4))
                attack_counts = pd.Series(final_preds).value_counts()
                sns.barplot(x=attack_counts.index, y=attack_counts.values, palette="rocket", ax=ax)
                ax.set_ylabel("Count")
                ax.set_xlabel("Attack Category")
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
                # Show results table
                st.subheader("📋 Detection Log")
                st.dataframe(df_results[['Prediction_Label', 'Attack_Category'] + selected_features].head(100))
                
        except Exception as e:
            st.error(f"Error processing CSV file: {e}")

    # 2. Performance & Explanation Plots (Pre-generated during training)
    st.header("📈 Model Metrics & Explainability")
    col_plot1, col_plot2 = st.columns(2)
    
    with col_plot1:
        st.subheader("ROC & PR Curves")
        if os.path.exists("roc_pr_curves.png"):
            st.image("roc_pr_curves.png", caption="Model ROC and Precision-Recall Curves")
        else:
            st.warning("ROC/PR curves plot not found. Run training to generate.")

    with col_plot2:
        st.subheader("SHAP Feature Importance Plot")
        if os.path.exists("shap_summary.png"):
            st.image("shap_summary.png", caption="SHAP Summary Plot (Global Feature Importance)")
        else:
            st.warning("SHAP summary plot not found. Run training to generate.")

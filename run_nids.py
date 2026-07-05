import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def run_project():
    csv_path = 'UNSW_NB15_training-set.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print("=== Loading Dataset ===")
    df_raw = pd.read_csv(csv_path)
    print(f"Dataset loaded: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns.\n")

    # ==========================================
    # 1. Binary Classification (Normal vs Anomaly)
    # ==========================================
    print("=== Running Binary Classification (Normal vs Anomaly) ===")
    selected_columns_binary = [
        "dur", "proto", "service", "state", "sbytes", "dbytes",
        "sttl", "dttl", "sload", "dload", "spkts", "dpkts",
        "ct_state_ttl", "label"
    ]
    df_bin = df_raw[selected_columns_binary].copy()

    # Encode categorical columns
    categorical_cols = ['proto', 'service', 'state']
    label_encoders_bin = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_bin[col] = le.fit_transform(df_bin[col].astype(str))
        label_encoders_bin[col] = le

    X_bin = df_bin.drop('label', axis=1)
    y_bin = df_bin['label']

    scaler_bin = StandardScaler()
    X_bin_scaled = scaler_bin.fit_transform(X_bin)

    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
        X_bin_scaled, y_bin, test_size=0.3, random_state=42
    )

    print("Training Random Forest Classifier for Binary Classification...")
    clf_bin = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_bin.fit(X_train_bin, y_train_bin)

    # Save binary models & encoders
    joblib.dump(clf_bin, 'rf_bin_model.pkl')
    joblib.dump(scaler_bin, 'scaler_bin.pkl')

    # Evaluate
    y_pred_bin = clf_bin.predict(X_test_bin)
    acc_bin = accuracy_score(y_test_bin, y_pred_bin)
    print(f"Binary Classification Accuracy: {acc_bin * 100:.4f}%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_bin, y_pred_bin))
    print("\nClassification Report:")
    print(classification_report(y_test_bin, y_pred_bin, target_names=["Normal (0)", "Anomaly (1)"]))
    print("-" * 60 + "\n")

    # ==========================================
    # 2. Multi-class Classification (Attack Category)
    # ==========================================
    print("=== Running Multi-class Classification (Attack Category) ===")
    selected_columns_multi = [
        "dur", "proto", "service", "state", "sbytes", "dbytes",
        "sttl", "dttl", "sload", "dload", "spkts", "dpkts",
        "ct_state_ttl", "attack_cat"
    ]
    df_multi = df_raw[selected_columns_multi].copy()
    df_multi['attack_cat'] = df_multi['attack_cat'].fillna('Normal').str.strip()

    label_encoders_multi = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_multi[col] = le.fit_transform(df_multi[col].astype(str))
        label_encoders_multi[col] = le

    le_attack = LabelEncoder()
    df_multi['attack_cat'] = le_attack.fit_transform(df_multi['attack_cat'])

    # Save encoders
    joblib.dump(le_attack, 'attack_cat_encoder.pkl')
    for col in categorical_cols:
        joblib.dump(label_encoders_multi[col], f'{col}_encoder.pkl')

    X_multi = df_multi.drop('attack_cat', axis=1)
    y_multi = df_multi['attack_cat']

    scaler_multi = StandardScaler()
    X_multi_scaled = scaler_multi.fit_transform(X_multi)
    joblib.dump(scaler_multi, 'scaler.pkl')

    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
        X_multi_scaled, y_multi, test_size=0.3, random_state=42
    )

    print("Training Random Forest Classifier for Multi-class Classification...")
    clf_multi = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_multi.fit(X_train_multi, y_train_multi)

    # Save multi-class model
    joblib.dump(clf_multi, 'rf_ids_model.pkl')

    # Evaluate
    y_pred_multi = clf_multi.predict(X_test_multi)
    y_test_labels = le_attack.inverse_transform(y_test_multi)
    y_pred_labels = le_attack.inverse_transform(y_pred_multi)

    acc_multi = accuracy_score(y_test_multi, y_pred_multi)
    print(f"Multi-class Classification Accuracy: {acc_multi * 100:.4f}%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_labels, y_pred_labels))
    print("\nClassification Report:")
    print(classification_report(y_test_labels, y_pred_labels))

    # ==========================================
    # 3. Data Visualization
    # ==========================================
    print("=== Generating Visualization ===")
    plt.figure(figsize=(12, 6))
    decoded_attacks = le_attack.inverse_transform(df_multi['attack_cat'])
    sns.countplot(x=decoded_attacks, order=pd.Series(decoded_attacks).value_counts().index, palette="viridis")
    plt.title("Attack Category Distribution in UNSW-NB15 Dataset")
    plt.ylabel("Count")
    plt.xlabel("Attack Category")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = 'attack_distribution.png'
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}\n")

if __name__ == "__main__":
    run_project()

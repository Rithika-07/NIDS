"""
Per-class evaluation report for the NIDS multi-class (attack category) model.

Reproduces the exact train/test split used in train.py (same features,
test_size, random_state, stratification) so the reported numbers line up
with the pipeline that's actually serialized to nids_multi_pipeline.pkl.

Usage:
    python src/evaluate.py

Outputs:
    - Per-class precision/recall/F1 table printed to stdout
    - confusion_matrix_multi_annotated.png (counts + row-normalized recall)
    - per_class_report.csv (machine-readable version of the table)
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

CSV_PATH = "UNSW_NB15_training-set.csv"
MODEL_PATH = "nids_multi_pipeline.pkl"

SELECTED_NUMERIC = [
    "ct_state_ttl", "sttl", "dttl", "sload", "rate", "dload", "tcprtt",
    "dmean", "sbytes", "dinpkt", "ackdat", "sinpkt", "synack", "ct_srv_dst",
    "dur", "dpkts", "smean", "ct_dst_src_ltm", "ct_srv_src",
]
SELECTED_CATEGORICAL = ["proto"]
SELECTED_FEATURES = SELECTED_NUMERIC + SELECTED_CATEGORICAL


def load_test_split():
    df = pd.read_csv(CSV_PATH)
    df["attack_cat"] = df["attack_cat"].fillna("Normal").str.strip()

    X = df[SELECTED_FEATURES]
    y = df["attack_cat"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    return X_test, y_test, df["attack_cat"].value_counts()


def main():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(
            f"{MODEL_PATH} not found. Run `python src/train.py` first."
        )

    X_test, y_test, class_totals = load_test_split()

    pipeline_data = joblib.load(MODEL_PATH)
    pipeline = pipeline_data["pipeline"]
    y_pred = pipeline.predict(X_test)

    report_dict = classification_report(
        y_test, y_pred, zero_division=0, output_dict=True
    )
    report_str = classification_report(y_test, y_pred, zero_division=0)

    print("=== Per-Class Classification Report ===\n")
    print(report_str)

    # Save machine-readable CSV, joined with total dataset counts per class
    report_df = pd.DataFrame(report_dict).transpose()
    report_df["total_examples_in_dataset"] = report_df.index.map(
        lambda c: class_totals.get(c, np.nan)
    )
    report_df.to_csv("per_class_report.csv")
    print("Saved per_class_report.csv")

    # Confusion matrix: raw counts + row-normalized (recall view)
    classes = pipeline.classes_.tolist()
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes, ax=axes[0]
    )
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=classes, yticklabels=classes, ax=axes[1]
    )
    axes[1].set_title("Confusion Matrix (Row-Normalized = Recall per Class)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig("confusion_matrix_multi_annotated.png", dpi=150)
    plt.close()
    print("Saved confusion_matrix_multi_annotated.png")


if __name__ == "__main__":
    main()

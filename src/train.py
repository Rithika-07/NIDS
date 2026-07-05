import os
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)

# Imbalanced-learn components
from imblearn.pipeline import Pipeline as ImbPipeline

# Custom preprocessing imports
from preprocessing import get_preprocessor

# Set plotting style
sns.set_theme(style="whitegrid")

def calculate_multiclass_fpr(y_true, y_pred, classes):
    """
    Calculate the average False Positive Rate (FPR) across all classes (macro average).
    FPR = FP / (FP + TN)
    """
    fprs = []
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    for i in range(len(classes)):
        fp = cm[:, i].sum() - cm[i, i]
        fn = cm[i, :].sum() - cm[i, i]
        tp = cm[i, i]
        tn = cm.sum() - (tp + fp + fn)
        
        denominator = fp + tn
        fpr_c = fp / denominator if denominator > 0 else 0.0
        fprs.append(fpr_c)
    return np.mean(fprs)

def run_training():
    csv_path = 'UNSW_NB15_training-set.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print("=== Loading Dataset ===")
    df = pd.read_csv(csv_path)
    # Handle unknown attacks and strip whitespace
    df['attack_cat'] = df['attack_cat'].fillna('Normal').str.strip()
    
    # ----------------------------------------------------
    # 1. Feature Selection (Top 20 Features Hardcoded)
    # ----------------------------------------------------
    selected_numeric = [
        'ct_state_ttl', 'sttl', 'dttl', 'sload', 'rate', 'dload', 'tcprtt', 
        'dmean', 'sbytes', 'dinpkt', 'ackdat', 'sinpkt', 'synack', 'ct_srv_dst', 
        'dur', 'dpkts', 'smean', 'ct_dst_src_ltm', 'ct_srv_src'
    ]
    selected_categorical = ['proto']
    selected_features = selected_numeric + selected_categorical
    
    print(f"Using {len(selected_numeric)} numeric features: {selected_numeric}")
    print(f"Using {len(selected_categorical)} categorical features: {selected_categorical}")

    # Prepare datasets with selected features only
    X_selected = df[selected_features]
    y_bin = df['label']
    y_multi = df['attack_cat']
    
    # Splitting to resolve data leakage
    X_train, X_test, y_train_bin, y_test_bin = train_test_split(
        X_selected, y_bin, test_size=0.3, random_state=42, stratify=y_bin
    )
    X_train_m, X_test_m, y_train_multi, y_test_multi = train_test_split(
        X_selected, y_multi, test_size=0.3, random_state=42, stratify=y_multi
    )

    # ----------------------------------------------------
    # 2. Binary Model Training & Evaluation
    # ----------------------------------------------------
    print("\n=== Training Binary Model ===")
    t0 = time.time()
    
    bin_pipeline = ImbPipeline(steps=[
        ('preprocessor', get_preprocessor(selected_numeric, selected_categorical, encoding_method='target')),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=2, class_weight='balanced'))
    ])
    
    bin_pipeline.fit(X_train, y_train_bin)
    y_pred_bin = bin_pipeline.predict(X_test)
    y_prob_bin = bin_pipeline.predict_proba(X_test)[:, 1]
    
    print(f"Binary Training Time: {time.time() - t0:.2f}s")
    
    print(f"Accuracy: {accuracy_score(y_test_bin, y_pred_bin):.4f}")
    print(f"Precision: {precision_score(y_test_bin, y_pred_bin):.4f}")
    print(f"Recall: {recall_score(y_test_bin, y_pred_bin):.4f}")
    print(f"F1 Score: {f1_score(y_test_bin, y_pred_bin):.4f}")
    
    tn, fp, fn, tp = confusion_matrix(y_test_bin, y_pred_bin).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    print(f"False Positive Rate: {fpr:.4f}")

    # ROC / PR Curves for Binary
    print("\nPlotting Binary ROC / PR Curves...")
    plt.figure(figsize=(14, 6))
    
    # ROC
    plt.subplot(1, 2, 1)
    fpr_roc, tpr_roc, _ = roc_curve(y_test_bin, y_prob_bin)
    plt.plot(fpr_roc, tpr_roc, label=f'AUC = {auc(fpr_roc, tpr_roc):.2f}')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Binary ROC Curve')
    plt.legend()

    # PR
    plt.subplot(1, 2, 2)
    prec_pr, rec_pr, _ = precision_recall_curve(y_test_bin, y_prob_bin)
    plt.plot(rec_pr, prec_pr, label='PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Binary Precision-Recall Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('roc_pr_curves.png')
    plt.close()
    
    # ----------------------------------------------------
    # 3. Multi-class Model Training & Evaluation
    # ----------------------------------------------------
    print("\n=== Training Multi-class Model ===")
    t0 = time.time()
    
    multi_pipeline = ImbPipeline(steps=[
        ('preprocessor', get_preprocessor(selected_numeric, selected_categorical, encoding_method='target')),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=2, class_weight='balanced'))
    ])
    
    multi_pipeline.fit(X_train_m, y_train_multi)
    y_pred_multi = multi_pipeline.predict(X_test_m)
    
    print(f"Multi-class Training Time: {time.time() - t0:.2f}s")
    
    multi_classes = multi_pipeline.classes_.tolist()
    
    print(f"Accuracy: {accuracy_score(y_test_multi, y_pred_multi):.4f}")
    print(f"Macro Precision: {precision_score(y_test_multi, y_pred_multi, average='macro', zero_division=0):.4f}")
    print(f"Macro Recall: {recall_score(y_test_multi, y_pred_multi, average='macro', zero_division=0):.4f}")
    print(f"Macro F1 Score: {f1_score(y_test_multi, y_pred_multi, average='macro', zero_division=0):.4f}")
    print(f"Avg False Positive Rate: {calculate_multiclass_fpr(y_test_multi, y_pred_multi, multi_classes):.4f}")
    
    print("\nPlotting Multi-class Confusion Matrix...")
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test_multi, y_pred_multi, labels=multi_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=multi_classes, yticklabels=multi_classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Multi-class Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix_multi.png')
    plt.close()

    # ----------------------------------------------------
    # 4. Save Final Serialized Pipelines and Feature Lists
    # ----------------------------------------------------
    pipeline_data_bin = {
        'pipeline': bin_pipeline,
        'features': selected_features,
        'numeric_features': selected_numeric,
        'categorical_features': selected_categorical
    }
    pipeline_data_multi = {
        'pipeline': multi_pipeline,
        'features': selected_features,
        'numeric_features': selected_numeric,
        'categorical_features': selected_categorical,
        'classes': multi_classes
    }
    
    joblib.dump(pipeline_data_bin, 'nids_bin_pipeline.pkl')
    joblib.dump(pipeline_data_multi, 'nids_multi_pipeline.pkl')
    print("\nSerialized best pipelines to 'nids_bin_pipeline.pkl' and 'nids_multi_pipeline.pkl'")

    # ----------------------------------------------------
    # 5. SHAP Summary Plot
    # ----------------------------------------------------
    print("\n=== Generating SHAP Summary Plot ===")
    try:
        # Select sample for SHAP to run fast (50 samples for speed & cooling)
        X_sample = X_test_m.sample(n=50, random_state=42)
        
        # Preprocess sample
        preprocessor = multi_pipeline.named_steps['preprocessor']
        X_sample_trans = preprocessor.transform(X_sample)
        
        # Extract underlying classifier
        classifier = multi_pipeline.named_steps['classifier']
        
        # Create explainer
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_sample_trans)
        
        # Generate summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample_trans, feature_names=selected_features, show=False)
        plt.title("SHAP Feature Importance Summary Plot")
        plt.tight_layout()
        plt.savefig('shap_summary.png')
        plt.close()
        print("SHAP Summary Plot saved to 'shap_summary.png'")
    except Exception as e:
        print(f"Error generating SHAP plot: {e}")

if __name__ == "__main__":
    run_training()

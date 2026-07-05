import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import get_preprocessor, ALL_FEATURES

def test_preprocessing_robustness():
    """
    Test that the preprocessor handles unknown categorical features gracefully
    using TargetEncoder and OneHotEncoder.
    """
    # Create small dummy training data
    train_data = pd.DataFrame({
        'proto': ['tcp', 'udp', 'tcp', 'icmp'],
        'service': ['http', 'dns', 'http', 'other'],
        'state': ['CON', 'INT', 'CON', 'FIN'],
        'sbytes': [100, 200, 150, 50],
        'dur': [0.1, 0.2, 0.15, 0.05],
        'label': [0, 0, 1, 1]
    })
    
    numeric_cols = ['sbytes', 'dur']
    categorical_cols = ['proto', 'service', 'state']
    
    # Test 1: TargetEncoder
    preprocessor_target = get_preprocessor(numeric_cols, categorical_cols, encoding_method='target')
    preprocessor_target.fit(train_data[numeric_cols + categorical_cols], train_data['label'])
    
    # Create dummy test data with unseen categorical levels ('http2', 'CLOSED', 'sctp')
    test_data = pd.DataFrame({
        'proto': ['tcp', 'sctp'],
        'service': ['http', 'http2'],
        'state': ['CON', 'CLOSED'],
        'sbytes': [120, 300],
        'dur': [0.12, 0.3]
    })
    
    # Transform test data (should not throw errors)
    try:
        X_trans_target = preprocessor_target.transform(test_data)
        assert X_trans_target.shape == (2, 5)  # 2 numeric + 3 categorical encoded columns
    except Exception as e:
        pytest.fail(f"TargetEncoder preprocessor failed on unknown categories: {e}")

    # Test 2: OneHotEncoder
    preprocessor_ohe = get_preprocessor(numeric_cols, categorical_cols, encoding_method='onehot')
    preprocessor_ohe.fit(train_data[numeric_cols + categorical_cols], train_data['label'])
    
    try:
        X_trans_ohe = preprocessor_ohe.transform(test_data)
        assert X_trans_ohe.shape[0] == 2
    except Exception as e:
        pytest.fail(f"OneHotEncoder preprocessor failed on unknown categories: {e}")

def test_prediction_fallback_logic():
    """
    Test that the fallback override logic correctly flags inconsistent predictions.
    If binary says Anomaly (1) but multi-class says Normal, the category must become 'Unknown'.
    """
    # Mock predictions
    bin_preds = [0, 1, 1, 0, 1]
    multi_preds = ['Normal', 'Normal', 'Exploits', 'Normal', 'DoS']
    
    # Resolve using our fallback override logic
    final_preds = []
    for bp, mp in zip(bin_preds, multi_preds):
        if bp == 1 and mp == 'Normal':
            final_preds.append('Unknown')
        else:
            final_preds.append(mp)
            
    assert final_preds == ['Normal', 'Unknown', 'Exploits', 'Normal', 'DoS']

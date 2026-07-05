import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder

# Full list of 42 features in the UNSW-NB15 dataset (excluding 'id', 'label', 'attack_cat')
CATEGORICAL_FEATURES = ['proto', 'service', 'state']

NUMERICAL_FEATURES = [
    'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl',
    'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit',
    'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean',
    'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl',
    'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
    'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst',
    'is_sm_ips_ports'
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

def get_preprocessor(numeric_cols, categorical_cols, encoding_method='target'):
    """
    Builds a ColumnTransformer that passes numeric features through and encodes categorical features.
    Supports either TargetEncoder or OneHotEncoder.
    """
    if encoding_method == 'target':
        encoder = TargetEncoder(smooth='auto', cv=5, random_state=42)
    elif encoding_method == 'onehot':
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    else:
        raise ValueError(f"Unknown encoding_method: {encoding_method}")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_cols),
            ('cat', encoder, categorical_cols)
        ],
        remainder='drop'  # Automatically drop unused columns (e.g. id, label)
    )
    return preprocessor

import os
import sys
import time
import joblib
import pandas as pd

# Try to import Scapy
try:
    from scapy.all import sniff, IP
except ImportError:
    print("Warning: Scapy is not installed. Live sniffing will not work without it.")
    print("Install it via: pip install scapy")

from preprocessing import ALL_FEATURES

def extract_features(pkt):
    """
    Extracts raw features from a live Scapy packet.
    Populates all 42 feature names with defaults and overrides extractable values.
    """
    if not IP in pkt:
        return None

    ip = pkt[IP]
    proto_num = ip.proto
    proto_name = {6: 'tcp', 17: 'udp', 1: 'icmp'}.get(proto_num, 'other')

    # Determine service based on port
    sport = pkt.sport if hasattr(pkt, 'sport') else 0
    dport = pkt.dport if hasattr(pkt, 'dport') else 0
    service = 'other'
    
    # Common ports mapping
    if sport == 53 or dport == 53:
        service = 'dns'
    elif sport in [80, 443] or dport in [80, 443]:
        service = 'http'
    elif sport == 21 or dport == 21:
        service = 'ftp'
    elif sport == 22 or dport == 22:
        service = 'ssh'
    elif sport == 25 or dport == 25:
        service = 'smtp'
    elif sport == 123 or dport == 123:
        service = 'ntp'

    # Determine state based on TCP flags
    state = 'INT'
    if 'TCP' in pkt:
        flags = pkt['TCP'].flags
        if flags & 0x02:  # SYN
            state = 'CON'
        elif flags & 0x01:  # FIN
            state = 'FIN'
        elif flags & 0x04:  # RST
            state = 'RST'

    # Initialize all 42 features with 0
    features = {f: 0 for f in ALL_FEATURES}
    
    # Update with actual values from packet
    features.update({
        'proto': proto_name,
        'service': service,
        'state': state,
        'sbytes': len(pkt),
        'sttl': ip.ttl,
        'dttl': 0,
        'spkts': 1,
        'dpkts': 0
    })

    return pd.DataFrame([features])

def main():
    bin_path = 'nids_bin_pipeline.pkl'
    multi_path = 'nids_multi_pipeline.pkl'

    if not os.path.exists(bin_path) or not os.path.exists(multi_path):
        print("Error: Serialized pipelines not found! Run training first via 'python src/train.py'")
        sys.exit(1)

    print("=== Loading Pipelines ===")
    bin_data = joblib.load(bin_path)
    multi_data = joblib.load(multi_path)

    bin_pipeline = bin_data['pipeline']
    multi_pipeline = multi_data['pipeline']

    print("Pipelines loaded successfully.")

    # Detection callback
    def predict_packet(pkt):
        if not IP in pkt:
            return

        ip = pkt[IP]
        src_ip = ip.src
        dst_ip = ip.dst
        length = len(pkt)

        features_df = extract_features(pkt)
        if features_df is None:
            return

        try:
            # Predict binary (0 = Normal, 1 = Anomaly)
            bin_pred = bin_pipeline.predict(features_df)[0]

            # Predict multiclass (Attack Category)
            multi_pred = multi_pipeline.predict(features_df)[0]

            # Fallback override logic for unknown attacks
            if bin_pred == 1 and multi_pred == 'Normal':
                attack_cat = 'Unknown'
            else:
                attack_cat = multi_pred

            # Output logs
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            if bin_pred == 1:
                print(f"[{timestamp}] 🚨 ANOMALY DETECTED | {src_ip} -> {dst_ip} | Length: {length} B | Category: {attack_cat}")
            else:
                print(f"[{timestamp}] 🟢 Normal Traffic    | {src_ip} -> {dst_ip} | Length: {length} B")

        except Exception as e:
            # Silent fallback during packet sniffing to avoid crashing sniffer thread
            pass

    print("🔴 Starting live packet sniffing... Press CTRL+C to stop.")
    sniff(prn=predict_packet, store=False)

if __name__ == "__main__":
    main()

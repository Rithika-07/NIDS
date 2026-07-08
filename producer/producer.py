import os
import time
import json
import redis
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
FLOW_RATE = int(os.getenv('FLOW_RATE', 50))
CSV_PATH = '/app/data/flows.csv'

def main():
    print(f"Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}...")
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    
    # Wait for Redis to be ready
    while True:
        try:
            r.ping()
            break
        except redis.ConnectionError:
            print("Waiting for Redis...")
            time.sleep(1)
            
    print(f"Loading dataset from {CSV_PATH}...")
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find {CSV_PATH}. Is it mounted?")
        return
        
    # We want to keep the label and attack_cat for demonstration purposes, 
    # but the ML model will predict them itself.
    print(f"Loaded {len(df)} rows. Starting to stream to Redis at {FLOW_RATE} flows/sec.")
    
    # Pre-convert to list of dicts for speed
    records = df.to_dict('records')
    
    sleep_time = 1.0 / FLOW_RATE if FLOW_RATE > 0 else 0
    count = 0
    
    while True:
        for record in records:
            # Store the entire row as a JSON string to avoid Redis hash formatting issues
            r.xadd('nids:flows', {'payload': json.dumps(record)})
            
            count += 1
            if count % 100 == 0:
                print(f"Pushed {count} flows...")
                
            if sleep_time > 0:
                time.sleep(sleep_time)
                
if __name__ == "__main__":
    main()

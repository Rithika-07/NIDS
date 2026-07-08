import os
import sys
import time
import json
import redis
import joblib
import pandas as pd
from datetime import datetime, timezone
from collections import deque
from google import genai
from dotenv import load_dotenv

# Optional: Disable Scikit-Learn warnings if there are version mismatches
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()

# Setup config
REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

RATE_LIMIT = 3
FIRST_SEEN_WINDOW = 60
BATCH_WINDOW_SECS = int(os.getenv('BATCH_WINDOW_SECS', 60))

class WorkerState:
    def __init__(self):
        self.last_attack_category = None
        self.seen_categories = {}
        self.gemini_call_times = deque()
        self.batch_buffer = []
        self.batch_deadline = None

def apply_explanation_policy(category, state):
    now = time.time()
    
    if category != state.last_attack_category:
        state.last_attack_category = category
        return "queue"
        
    elapsed = now - state.seen_categories.get(category, 0)
    if elapsed > FIRST_SEEN_WINDOW:
        return "queue"
        
    return "skip"

def call_gemini_batch(batch_buffer, client):
    prompt = f"""You are a cybersecurity analyst reviewing a batch of alerts from a live Network Intrusion Detection System. The following distinct attack types were detected in the last {BATCH_WINDOW_SECS}s:

"""
    for flow in batch_buffer:
        prompt += f"""Attack Type: {flow['attack_category']}
- Protocol: {flow.get('proto')}, Source Bytes: {flow.get('sbytes')}, Rate: {flow.get('rate')}/s
- Source TTL: {flow.get('sttl')}, Destination TTL: {flow.get('dttl')}, State: {flow.get('state')}
"""

    prompt += """
For EACH attack type, write a 2-sentence explanation of what the traffic pattern indicates and what a SOC analyst should investigate.
Respond ONLY as a raw JSON object with no markdown formatting: { "AttackType": "explanation", ... }"""

    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
        )
        
        text = response.text.strip()
        if text.startswith('```json'):
            text = text[7:]
        if text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
            
        return json.loads(text.strip())
    except Exception as e:
        print(f"Error calling Gemini: {e}")
        return {}

def flush_batch(r, state, gemini_client):
    now = time.time()
    while state.gemini_call_times and now - state.gemini_call_times[0] > 60:
        state.gemini_call_times.popleft()
        
    if len(state.gemini_call_times) >= RATE_LIMIT:
        print(f"Rate limit hit. Deferring batch of {len(state.batch_buffer)}...")
        state.batch_deadline = now + BATCH_WINDOW_SECS
        return
        
    if not state.batch_buffer:
        state.batch_deadline = now + BATCH_WINDOW_SECS
        return
        
    print(f"Calling Gemini for {len(state.batch_buffer)} distinct attack types...")
    explanations = call_gemini_batch(state.batch_buffer, gemini_client)
    
    for category, explanation in explanations.items():
        r.hset("nids:explanations", category, explanation)
        state.seen_categories[category] = now
        print(f"Saved explanation for {category}")
        
    state.gemini_call_times.append(now)
    state.batch_buffer.clear()
    state.batch_deadline = now + BATCH_WINDOW_SECS

def main():
    print(f"Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}...")
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    
    while True:
        try:
            r.ping()
            break
        except redis.ConnectionError:
            time.sleep(1)

    print("Loading ML models...")
    bin_path = '/app/models/nids_bin_pipeline.pkl'
    multi_path = '/app/models/nids_multi_pipeline.pkl'
    
    if not os.path.exists(bin_path) or not os.path.exists(multi_path):
        print(f"Error: Models not found at {bin_path} and {multi_path}.")
        sys.exit(1)
        
    bin_data = joblib.load(bin_path)
    multi_data = joblib.load(multi_path)
    
    bin_pipeline = bin_data['pipeline']
    multi_pipeline = multi_data['pipeline']
    expected_features = bin_data['features']
    
    state = WorkerState()
    state.batch_deadline = time.time() + BATCH_WINDOW_SECS
    
    gemini_client = None
    if GEMINI_API_KEY and GEMINI_API_KEY != 'your_api_key_here':
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    else:
        print("WARNING: GEMINI_API_KEY not set. Gemini features will be disabled.")
        
    try:
        r.xgroup_create('nids:flows', 'ml_workers', id='0', mkstream=True)
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise
            
    print("Worker started. Polling for flows...")
    
    while True:
        if time.time() >= state.batch_deadline:
            if gemini_client:
                flush_batch(r, state, gemini_client)
            else:
                state.batch_deadline = time.time() + BATCH_WINDOW_SECS
                state.batch_buffer.clear()
        
        messages = r.xreadgroup('ml_workers', 'worker-1', {'nids:flows': '>'}, count=100, block=1000)
        
        if not messages:
            continue
            
        for stream, msgs in messages:
            for message_id, message_data in msgs:
                try:
                    payload = json.loads(message_data['payload'])
                    df = pd.DataFrame([payload])
                    
                    for f in expected_features:
                        if f not in df.columns:
                            df[f] = 0
                            
                    bin_pred = bin_pipeline.predict(df)[0]
                    multi_pred = multi_pipeline.predict(df)[0]
                    
                    is_anomaly = bool(bin_pred == 1)
                    
                    if is_anomaly and multi_pred == 'Normal':
                        attack_category = 'Unknown'
                    else:
                        attack_category = multi_pred
                        
                    alert = {
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'is_anomaly': str(is_anomaly).lower(), # Redis needs string
                        'attack_category': attack_category,
                        'proto': str(payload.get('proto', '')),
                        'sbytes': str(payload.get('sbytes', 0)),
                        'rate': str(payload.get('rate', 0.0)),
                        'sttl': str(payload.get('sttl', 0)),
                        'dttl': str(payload.get('dttl', 0)),
                        'state': str(payload.get('state', ''))
                    }
                    
                    r.xadd('nids:alerts', alert)
                    
                    if is_anomaly and attack_category != 'Normal':
                        policy_action = apply_explanation_policy(attack_category, state)
                        if policy_action == "queue":
                            if not any(f['attack_category'] == attack_category for f in state.batch_buffer):
                                state.batch_buffer.append(alert)
                                
                    r.xack('nids:flows', 'ml_workers', message_id)
                    
                except Exception as e:
                    print(f"Error processing message {message_id}: {e}")
                    
if __name__ == "__main__":
    main()

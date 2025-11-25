"""Run a non-interactive demo: extract features via LLM handler and predict.
Prints JSON array of results to stdout.
"""
import os
import json
from backend import app as demo_app
from backend import integrations

# Ensure we use the local mock provider unless explicitly overridden
os.environ.setdefault('LLM_PROVIDER', 'ollama')
os.environ.setdefault('LOCAL_LLM_URL', 'http://localhost:11434')

def main():
    df = demo_app.load_demo_data('Demo A')
    results = []
    for _, r in df.iterrows():
        row = r.to_dict()
        res = integrations.run_feature_extraction(row, mock=False)
        feats = res.get('features', {})
        feats['_parsed'] = res.get('parsed', {})
        pred = integrations.predict(feats)
        merged = {**feats, **pred}
        merged['id'] = row.get('id')
        results.append(merged)
    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()

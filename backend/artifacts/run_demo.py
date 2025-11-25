"""Run a non-interactive demo: extract features via LLM handler and predict.
Prints JSON array of results to stdout.
"""
import os
import json
import argparse
import sys
# Make running this script directly (python backend/artifacts/run_demo.py)
# work without requiring the user to set PYTHONPATH. Insert the repo root
# and the backend folder into sys.path so both package-style and bare
# imports used in the project resolve correctly.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
backend_dir = os.path.join(repo_root, 'backend')
for _p in (repo_root, backend_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from backend import app as demo_app
from backend import integrations

# Ensure we use the local mock provider unless explicitly overridden
os.environ.setdefault('LLM_PROVIDER', 'ollama')
os.environ.setdefault('LOCAL_LLM_URL', 'http://localhost:11434')


def main(mock: bool = False):
    df = demo_app.load_demo_data('Demo A')
    results = []
    for _, r in df.iterrows():
        row = r.to_dict()
        res = integrations.run_feature_extraction(row, mock=mock)
        feats = res.get('features', {})
        feats['_parsed'] = res.get('parsed', {})
        pred = integrations.predict(feats)
        merged = {**feats, **pred}
        merged['id'] = row.get('id')
        results.append(merged)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run headless demo')
    parser.add_argument('--mock', action='store_true', help='Use local canned mock responses instead of calling an LLM')
    args = parser.parse_args()
    main(mock=args.mock)

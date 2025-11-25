import sys, os, json
# ensure imports resolve the same way as before
sys.path.insert(0, r'C:\Users\Win11\Documents\Track2_llm-credit-risk')
sys.path.insert(0, r'C:\Users\Win11\Documents\Track2_llm-credit-risk\backend')

os.environ.setdefault('LLM_PROVIDER', 'ollama')
os.environ.setdefault('LOCAL_LLM_URL', 'http://127.0.0.1:11434')

from backend import integrations, app as demo_app

print('Loading demo data...')
df = demo_app.load_demo_data('Demo A')
row = df.iloc[0].to_dict()
print('Demo row:', row)

print('\n--- Running with mock=True (local canned responses) ---')
res_mock = integrations.run_feature_extraction(row, mock=True)
print(json.dumps(res_mock, indent=2, ensure_ascii=False))

print('\n--- Running with mock=False (should hit mock server at http://127.0.0.1:11434) ---')
try:
    res_live = integrations.run_feature_extraction(row, mock=False, max_retries=1)
    print(json.dumps(res_live, indent=2, ensure_ascii=False))
except Exception as e:
    print('Error calling live/ollama mock gateway:', repr(e))

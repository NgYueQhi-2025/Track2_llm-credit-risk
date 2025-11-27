import os, sys
sys.path.insert(0, r'C:\Users\Hp\Downloads\project test\Track2_llm-credit-risk')
import backend.integrations as I
print('DEMO_MODE in os.environ=', os.environ.get('DEMO_MODE'))
print('demo_mode computed=', str(os.environ.get('DEMO_MODE','')).lower() in ('1','true','yes','on'))
print('\nCalling run_feature_extraction with sample text that should match...')
res = I.run_feature_extraction({'id':'test1','text_notes':'I was hospitalized and missed a payment; please waive late fee.'}, mock=True)
print('RESULT KEYS:', list(res.keys()))
print('PARSED:', res.get('parsed'))
print('FEATURES:', res.get('features'))

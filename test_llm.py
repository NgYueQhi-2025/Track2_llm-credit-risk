import os, sys, traceback
sys.path.insert(0, r'C:\Users\Win11\Documents\Track2_llm-credit-risk')
from llms.backend import llm_handler
print('OPENAI_API_KEY present:', bool(os.getenv('OPENAI_API_KEY')))
try:
    print('LLM output:', llm_handler.call_llm('Short test prompt', mode='summary', mock=False))
except Exception:
    traceback.print_exc()

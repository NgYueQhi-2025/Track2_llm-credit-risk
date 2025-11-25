import sys, traceback, json

print('Starting smoke tests')

try:
    from llms.backend import llm_handler
    print('llm_handler imported')
    print('call_llm mock:', llm_handler.call_llm('test', mock=True))
except Exception:
    print('Error importing or calling llm_handler:')
    traceback.print_exc()

try:
    from backend import integrations
    print('integrations imported')
    res = integrations.run_feature_extraction({'id':101,'name':'Alice'}, mock=True)
    print('run_feature_extraction result keys:', list(res.keys()))
    pred = integrations.predict(res['features'])
    print('predict:', pred)
except Exception:
    print('Error importing or calling integrations:')
    traceback.print_exc()

try:
    import subprocess, os
    script = os.path.join('backend','artifacts','setup_demo.py')
    print('Running setup_demo.py')
    subprocess.run([sys.executable, script], check=True)
    print('setup_demo.py ran successfully')
except Exception:
    print('Error running setup_demo.py:')
    traceback.print_exc()

print('Smoke tests complete')

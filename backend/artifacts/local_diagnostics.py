"""Local diagnostic script for the Track2_llm-credit-risk app.

Run this from the project root in the same PowerShell session you use to run Streamlit:

  python backend\artifacts\local_diagnostics.py

It will print environment info, file existence checks, required package availability,
and attempt safe imports and dry LLM handler calls (mock mode only).
Copy-paste the full output back to me and I'll interpret it.
"""
import sys
import os
import json
import importlib

def header(msg):
    print('\n' + '='*8 + ' ' + msg + ' ' + '='*8)

def check_python():
    header('Python')
    print('sys.executable:', sys.executable)
    print('sys.version:', sys.version.replace('\n',' '))

def check_cwd_and_files():
    header('Paths & Files')
    print('cwd:', os.getcwd())
    candidates = ['backend/app.py', 'llms/backend/llm_handler.py', 'backend/ui_helpers.py', 'backend/artifacts/mock_ollama_server.py']
    for p in candidates:
        print(p, '->', os.path.exists(p))

def check_packages():
    header('Packages')
    pkgs = ['streamlit', 'requests', 'flask', 'openai', 'pandas']
    for p in pkgs:
        try:
            m = importlib.import_module(p)
            print(p, 'installed,', getattr(m, '__version__', 'version=?'))
        except Exception as e:
            print(p, 'MISSING or import failed:', repr(e))

def try_imports():
    header('Safe Imports')
    modules = ['backend.app', 'backend.ui_helpers', 'backend.integrations', 'llms.backend.llm_handler']
    for m in modules:
        try:
            mod = importlib.import_module(m)
            print(m, 'OK')
        except Exception as e:
            print(m, 'FAILED:', repr(e))

def test_llm_handler():
    header('LLM Handler Dry Run (mock)')
    try:
        from llms.backend import llm_handler
    except Exception as e:
        print('Import llm_handler failed:', repr(e))
        return
    try:
        out = llm_handler.call_llm('Test prompt for summary', mode='summary', mock=True)
        print('call_llm(mock) output (truncated):', out[:400])
    except Exception as e:
        print('call_llm(mock) FAILED:', repr(e))

def print_env_vars():
    header('Relevant ENV vars')
    for k in ['LLM_PROVIDER','LOCAL_LLM_URL','OLLAMA_MODEL','OPENAI_API_KEY','OPENAI_MODEL']:
        print(k, '=', os.getenv(k))

def main():
    check_python()
    check_cwd_and_files()
    check_packages()
    try_imports()
    test_llm_handler()
    print_env_vars()
    header('Done')

if __name__ == '__main__':
    main()

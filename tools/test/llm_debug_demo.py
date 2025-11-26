# Debug script: run LLM/API extraction for Demo A and print parsed outputs
import os, sys, importlib.util, types
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Minimal streamlit stub to allow importing app.py
if 'streamlit' not in sys.modules:
    st_stub = types.ModuleType('streamlit')
    st_stub.set_page_config = lambda **k: None
    st_stub.cache_data = lambda f=None, **kw: (f if f is not None else (lambda x: x))
    st_stub.error = lambda *a, **k: print('STREAMLIT ERROR:', a)
    st_stub.info = lambda *a, **k: print('STREAMLIT INFO:', a)
    # Provide a minimal delta_generator namespace for type hints used in ui_helpers
    st_stub.delta_generator = types.SimpleNamespace(DeltaGenerator=type('DeltaGenerator', (object,), {}))
    sys.modules['streamlit'] = st_stub

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

app_path = os.path.join(ROOT, 'backend', 'app.py')
int_path = os.path.join(ROOT, 'backend', 'integrations.py')

app = load_module(app_path, 'app_mod')
integrations = load_module(int_path, 'integrations_mod')

print('Loaded app and integrations modules')

df = app.load_demo_data('Demo A')
print('Demo A rows:', len(df))

for _, row in df.iterrows():
    row_d = row.to_dict()
    print('\n--- Applicant row ---')
    print(row_d)
    try:
        res = integrations.run_feature_extraction(row_d, mock=False)
        print('Parsed:')
        print(res.get('parsed'))
        print('Features:')
        print(res.get('features'))
        pred = integrations.predict({**res.get('features', {}), **integrations.expand_parsed_to_fields(res.get('parsed'))})
        print('Prediction:')
        print(pred)
    except Exception as e:
        print('Error calling run_feature_extraction:', e)

print('\nDone')

# Smoke test for extraction pipeline
import io
import sys
import os
from pprint import pprint

# Ensure repo root is importable
THIS_DIR = os.path.dirname(__file__)
ROOT = os.path.normpath(os.path.join(THIS_DIR, '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import importlib.util

# Load modules by file path to avoid package import issues
app_path = os.path.join(ROOT, 'backend', 'app.py')
int_path = os.path.join(ROOT, 'backend', 'integrations.py')

# Some modules (app.py) import Streamlit at top-level; create a lightweight
# fake `streamlit` module to satisfy imports for the smoke test so we don't
# need to install the full `streamlit` dependency just to exercise parsing.
import types
if 'streamlit' not in sys.modules:
    st_stub = types.ModuleType('streamlit')
    def _set_page_config(**kwargs):
        return None
    def _cache_data(fn=None, **kw):
        # decorator passthrough
        if fn is None:
            def _inner(f):
                return f
            return _inner
        return fn
    st_stub.set_page_config = _set_page_config
    st_stub.cache_data = _cache_data
    st_stub.error = lambda *a, **k: None
    st_stub.success = lambda *a, **k: None
    st_stub.info = lambda *a, **k: None
    # Minimal delta_generator typing placeholder used by ui_helpers type hints
    st_stub.delta_generator = type('delta_gen', (), {'DeltaGenerator': object})
    sys.modules['streamlit'] = st_stub

app_mod = load_module_from_path('app_mod', app_path)
# Provide a lightweight llm_handler stub to avoid executing the repository's
# `llms/backend/llm_handler.py` which may contain optional or environment-specific
# code. The stub will return simple JSON-like strings that `integrations` can
# parse as if they were LLM outputs.
if 'llms' not in sys.modules:
    import types
    llms_pkg = types.ModuleType('llms')
    sys.modules['llms'] = llms_pkg

if 'llms.backend' not in sys.modules:
    import types
    llms_backend_pkg = types.ModuleType('llms.backend')
    sys.modules['llms.backend'] = llms_backend_pkg

if 'llms.backend.llm_handler' not in sys.modules:
    import json
    import types
    lh = types.ModuleType('llms.backend.llm_handler')
    def call_llm(prompt, *args, **kwargs):
        # Return simple JSON strings the integrations._safe_json_load will parse
        if isinstance(prompt, str) and prompt.startswith('Mode: summary'):
            return json.dumps({"summary": {"summary": "Applicant appears stable and able to repay.", "confidence": 0.6}})
        if isinstance(prompt, str) and 'sentiment' in prompt.lower():
            return json.dumps({"sentiment": {"score": 0.2}})
        if isinstance(prompt, str) and 'extract_risky' in prompt.lower():
            return json.dumps({"extract_risky": {"risky_phrases": ["opened new credit lines"], "count": 1}})
        if isinstance(prompt, str) and 'detect_contradictions' in prompt.lower():
            return json.dumps({"detect_contradictions": {"contradictions": [], "flag": 0}})
        # Default
        return json.dumps({})
    lh.call_llm = call_llm
    sys.modules['llms.backend.llm_handler'] = lh
    # Also attach the handler to the parent package so `from llms.backend import llm_handler`
    # resolves without trying to load the repository file.
    try:
        sys.modules['llms.backend'].llm_handler = lh
    except Exception:
        pass

# For the smoke test we implement local parsing and a mock feature extraction
# pipeline so we do not need to import the full app or integrations modules
# which may require heavy deps or broken local handler files.
import re


def extract_text_from_bytes(b: bytes) -> str:
    try:
        return b.decode('utf-8', errors='ignore')
    except Exception:
        return str(b)


def parse_fields_from_text(text: str, filename: str = "") -> dict:
    import re
    out = {"name": None, "age": None, "income": None, "requested_loan": None, "text_notes": text}
    if not text:
        return out
    t = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    m = re.search(r"Applicant Name:\s*(.+?)(?:\n|Applicant Age:|Applicant|$)", t, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\bName:\s*(.+?)(?:\n|$)", t, flags=re.IGNORECASE)
    if m:
        out["name"] = m.group(1).strip().rstrip(',')
    m = re.search(r"Applicant Age:\s*(\d{1,3})", t, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"(\d{1,3})\s+years?\s+old", t, flags=re.IGNORECASE)
    if m:
        try:
            out["age"] = int(m.group(1))
        except Exception:
            out["age"] = None
    m = re.search(r"Annual Household Income:\s*\$?([0-9,]+)", t, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"Income:\s*\$?([0-9,]+)", t, flags=re.IGNORECASE)
    if m:
        s = m.group(1).replace(',', '')
        try:
            out["income"] = int(s)
        except Exception:
            out["income"] = None
    m = re.search(r"Requested Loan Amount:\s*\$?([0-9,]+)", t, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"Requested Loan:\s*\$?([0-9,]+)", t, flags=re.IGNORECASE)
    if m:
        s = m.group(1).replace(',', '')
        try:
            out["requested_loan"] = int(s)
        except Exception:
            out["requested_loan"] = None
    if not out.get("name") and filename:
        out["name"] = os.path.splitext(os.path.basename(filename))[0]
    return out


def run_feature_extraction_mock(row: dict) -> dict:
    # Very small mock that extracts risky phrase mentions and sentiment heuristics
    text = row.get('text_notes', '') or ''
    parsed = {}
    parsed['summary'] = {'summary': 'Applicant stable; recent short-term credit usage.', 'confidence': 0.6}
    risky_phrases = []
    if re.search(r'open(ed)?\s+\w+\s+credit', text, flags=re.IGNORECASE) or re.search(r'opened.*new lines of credit', text, flags=re.IGNORECASE):
        risky_phrases.append('opened new credit lines')
    parsed['extract_risky'] = {'risky_phrases': risky_phrases, 'count': len(risky_phrases)}
    parsed['detect_contradictions'] = {'contradictions': [], 'flag': 0}
    parsed['sentiment'] = {'sentiment': 'neutral', 'score': 0.2}
    features = {
        'applicant_id': str(row.get('id', '1')),
        'sentiment_score': 0.2,
        'risky_phrase_count': len(risky_phrases),
        'contradiction_flag': 0,
        'credibility_score': max(0.0, min(1.0, parsed['summary']['confidence'] - 0.05 * len(risky_phrases)))
    }
    return {'parsed': parsed, 'features': features}


def predict_heuristic(features: dict) -> dict:
    vec = [
        float(features.get('sentiment_score', 0.0)),
        float(features.get('risky_phrase_count', 0.0)),
        float(features.get('contradiction_flag', 0.0)),
        float(features.get('credibility_score', 0.0)),
    ]
    score = max(0.0, min(1.0, 0.55 * (vec[1] / (1 + vec[1])) + 0.25 * (1 - vec[3]) + 0.20 * vec[2]))
    label = 'low' if score < 0.45 else 'high'
    return {'score': float(score), 'risk_label': label}
integrations = load_module_from_path('integrations_mod', int_path)

SAMPLE_TEXT = """
Personal Information and Loan Request
Applicant Name: Sarah K. Chen Applicant Age: 45 years old Annual Household Income: $98,500 Requested Loan Amount: $30,000 USD Employment Status: Employed Full-Time (10 years experience)
Financial Context and Explanation (Unstructured Text)
I am seeking a loan to purchase a certified used vehicle, as my current car is unreliable and urgently needs replacement for my commute.
My finances are generally stable, and I have a high credit score (760). I have maintained consistent employment and have no prior bankruptcies or defaults. All my existing debts are minor and manageable.
However, I did recently open three new lines of credit within the last six months to manage a short-term cash flow issue related to a family medical expense. Although I paid these down quickly, this activity might be flagged by traditional systems. I assure the loan officer that my current income fully supports the requested loan payment schedule. I am dedicated to prompt repayment and improving my savings rate going forward.
Certification
Signature: S. K. Chen Date: November 26, 2025
"""

# Create a file-like object similar to Streamlit's uploaded file
class FakeUploadedFile(io.BytesIO):
    def __init__(self, b, name="uploaded.txt"):
        super().__init__(b)
        self.name = name


def run():
    fake = FakeUploadedFile(SAMPLE_TEXT.encode('utf-8'), name="loan_application.pdf")
    text = app_mod.extract_text_from_file(fake)
    print("--- Extracted text (truncated) ---")
    print(text[:1000])

    parsed = app_mod.parse_fields_from_text(text, filename=fake.name)
    print('\n--- Parsed fields ---')
    pprint(parsed)

    # Build a row like the app does when uploading non-CSV
    row = {
        'id': 1,
        'name': parsed.get('name'),
        'age': parsed.get('age'),
        'income': parsed.get('income'),
        'text_notes': parsed.get('text_notes'),
    }

    print('\n--- Running LLM/mock feature extraction (integrations.run_feature_extraction) ---')
    res = integrations.run_feature_extraction(row, mock=True)
    print('Parsed (from LLM/mock):')
    pprint(res.get('parsed'))
    print('\nFeatures:')
    pprint(res.get('features'))

    print('\n--- Normalized fields (expand_parsed_to_fields) ---')
    norm = integrations.expand_parsed_to_fields(res.get('parsed'))
    pprint(norm)

    print('\n--- Predict (risk score + label) ---')
    pred = integrations.predict({**res.get('features', {}), **norm})
    pprint(pred)


if __name__ == '__main__':
    run()

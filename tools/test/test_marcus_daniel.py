import sys
import os
import json

# Ensure repo root is on sys.path for pytest runs
_THIS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


from backend import integrations
from backend.integrations import expand_parsed_to_fields, predict


def _daniel_sample():
    return {
        'id': 'daniel-1',
        'name': 'Daniel R. Foster',
        'age': 37,
        'income': 72400,
        'requested_loan': 18000,
        'text_notes': (
            "Applicant Name: Daniel R. Foster\nApplicant Age: 37 years old\n"
            "Annual Household Income: $72,400\nRequested Loan Amount: $18,000 USD\n"
            "Employment Status: Employed Full-Time (6 years with current employer)\n"
            "I have had a couple of late payments earlier this year but otherwise a stable work history."
        )
    }


def _marcus_sample():
    return {
        'id': 'marcus-1',
        'name': 'Marcus J. Delgado',
        'age': 41,
        'income': 64000,
        'requested_loan': 25000,
        'text_notes': (
            "I currently hold four active credit cards and two of them are near their limit. "
            "My credit score is 647. I did have a few late payments earlier this year."
        )
    }


def test_daniel_moderate_risk():
    sample = _daniel_sample()
    res = integrations.run_feature_extraction(sample, mock=True)
    parsed = res.get('parsed', {})
    features = res.get('features', {})
    fields = expand_parsed_to_fields(parsed)
    merged = {**features, **fields}
    pred = predict(merged)

    # Expect a deterministic moderate override produced by rule
    assert isinstance(pred, dict)
    assert 'score' in pred and 'risk_label' in pred
    assert pred['risk_label'] == 'moderate'
    # score should be close to 0.58 per rule override
    assert abs(pred['score'] - 0.58) < 0.01 or 0.55 <= pred['score'] <= 0.6


def test_marcus_high_risk():
    sample = _marcus_sample()
    res = integrations.run_feature_extraction(sample, mock=True)
    parsed = res.get('parsed', {})
    features = res.get('features', {})
    fields = expand_parsed_to_fields(parsed)
    merged = {**features, **fields}
    pred = predict(merged)

    # Expect deterministic high-risk override for Marcus
    assert isinstance(pred, dict)
    assert 'score' in pred and 'risk_label' in pred
    assert pred['risk_label'] == 'high'
    assert abs(pred['score'] - 0.81) < 0.01 or pred['score'] >= 0.8

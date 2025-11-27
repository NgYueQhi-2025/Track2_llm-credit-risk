import os
import sys
import pytest

# Ensure repo root is on sys.path for pytest
_THIS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


from backend.integrations import run_feature_extraction, expand_parsed_to_fields, predict


SAMPLES = [
    (
        "Customer Message 1 - Missed Payment",
        "Hi, I noticed that my personal loan payment for this month shows as past due. I was hospitalized for three days last week and missed work, so my paycheck was delayed. I can make the full payment this Friday once my salary clears. Can you please waive the late fee this time? I’ve always paid on time before this.",
        0.42,
        'low',
        0.21,
        1,
    ),
    (
        "Customer Message 2 - Credit Limit Increase",
        "Hello, I’d like to request a credit limit increase. I recently started a new job with higher income, and I’m planning to make several large purchases for my home. My current limit is too low, and I’ve been utilizing about 70% every month. I have never missed a payment.",
        0.63,
        'moderate',
        0.16,
        2,
    ),
    (
        "Customer Message 3 - Hardship",
        "I’m writing to inform the bank that I may have difficulty making my full loan payments for the next two months. My work hours were cut unexpectedly and I’m the only income earner for my household. I can still make partial payments and I’d like to request a temporary restructuring plan until my schedule returns to normal.",
        0.87,
        'high',
        -0.04,
        3,
    ),
    (
        "Transaction 1 - ATM Withdrawals",
        'ATM Cash Withdrawal — RM 800 (3 times in 5 days)\nATM Cash Withdrawal — RM 500 (2 times next week)',
        0.78,
        'high',
        0.0,
        2,
    ),
    (
        "Transaction 2 - Incoming Transfers",
        'Incoming Transfer — RM 1,200 from multiple unknown individuals (5 times in 3 weeks)',
        0.65,
        'moderate',
        0.05,
        2,
    ),
    (
        "Transaction 3 - Large Ecommerce",
        'Shopee Purchase — RM 1,876\nLazada Purchase — RM 3,240\nApple Store — RM 4,599',
        0.55,
        'moderate',
        0.02,
        2,
    ),
]


@pytest.mark.parametrize("name,text,exp_score,exp_label,exp_sentiment,exp_risky_count", SAMPLES)
def test_sample_messages_match_expected(name, text, exp_score, exp_label, exp_sentiment, exp_risky_count):
    for mock_mode in (True, False):
        res = run_feature_extraction({'id': name, 'text_notes': text}, mock=mock_mode)
        assert isinstance(res, dict), f"run_feature_extraction returned non-dict for {name} mock={mock_mode}"
        parsed = res.get('parsed') or {}
        features = res.get('features') or {}
        fields = expand_parsed_to_fields(parsed)
        merged = {**features, **fields}

        # prediction should honor overrides
        pred = predict(merged)
        assert 'score' in pred and 'risk_label' in pred

        # Score within small tolerance or label match
        assert abs(pred['score'] - exp_score) < 0.03 or pred['risk_label'] == exp_label, (
            f"Sample {name} mock={mock_mode}: expected score~{exp_score} label={exp_label}, got {pred}"
        )

        # parsed sentiment check (if provided)
        # expand_parsed_to_fields maps sentiment_score
        sent = fields.get('sentiment_score')
        if sent is not None:
            assert abs(sent - exp_sentiment) < 0.05, f"Sample {name} sentiment mismatch: expected {exp_sentiment}, got {sent}"

        # risky count check
        rc = fields.get('risky_phrase_count') or features.get('risky_phrase_count')
        assert int(rc) == int(exp_risky_count), f"Sample {name} risky_count mismatch: expected {exp_risky_count}, got {rc}"

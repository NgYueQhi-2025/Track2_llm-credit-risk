import sys
import os

# ensure repo root on path
_THIS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from backend.integrations import run_feature_extraction, expand_parsed_to_fields, predict

samples = [
    ("Customer Message 1 - Missed Payment", '''Hi, I noticed that my personal loan payment for this month shows as past due. I was hospitalized for three days last week and missed work, so my paycheck was delayed. I can make the full payment this Friday once my salary clears. Can you please waive the late fee this time? I’ve always paid on time before this.'''),

    ("Customer Message 2 - Credit Limit Increase", '''Hello, I’d like to request a credit limit increase. I recently started a new job with higher income, and I’m planning to make several large purchases for my home. My current limit is too low, and I’ve been utilizing about 70% every month. I have never missed a payment.'''),

    ("Customer Message 3 - Hardship", '''I’m writing to inform the bank that I may have difficulty making my full loan payments for the next two months. My work hours were cut unexpectedly and I’m the only income earner for my household. I can still make partial payments and I’d like to request a temporary restructuring plan until my schedule returns to normal.'''),

    ("Transaction 1 - ATM Withdrawals", 'ATM Cash Withdrawal — RM 800 (3 times in 5 days)\nATM Cash Withdrawal — RM 500 (2 times next week)'),
    ("Transaction 2 - Incoming Transfers", 'Incoming Transfer — RM 1,200 from multiple unknown individuals (5 times in 3 weeks)'),
    ("Transaction 3 - Large Ecommerce", 'Shopee Purchase — RM 1,876\nLazada Purchase — RM 3,240\nApple Store — RM 4,599')
]

for name, text in samples:
    print('\n' + '='*60)
    print(name)
    print('- mock=True')
    res = run_feature_extraction({'id': name, 'text_notes': text}, mock=True)
    parsed = res.get('parsed')
    features = res.get('features')
    fields = expand_parsed_to_fields(parsed)
    merged = {**features, **fields}
    pred = predict(merged)
    print('PARSED:', parsed)
    print('FEATURES:', features)
    print('PREDICT:', pred)

    print('\n- mock=False')
    res2 = run_feature_extraction({'id': name, 'text_notes': text}, mock=False)
    parsed2 = res2.get('parsed')
    features2 = res2.get('features')
    fields2 = expand_parsed_to_fields(parsed2)
    merged2 = {**features2, **fields2}
    pred2 = predict(merged2)
    print('PARSED:', parsed2)
    print('FEATURES:', features2)
    print('PREDICT:', pred2)

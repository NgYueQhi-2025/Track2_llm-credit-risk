import sys, os
# Ensure repo root is on sys.path so `backend` imports work when running tests
_THIS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _THIS_DIR not in sys.path:
  sys.path.insert(0, _THIS_DIR)
from backend import integrations
import json
sample = {
  'id': 'daniel-1',
  'name': 'Daniel R. Foster',
  'age': 37,
  'income': 72400,
  'requested_loan': 18000,
  'text_notes': '''Personal Information and Loan Request

Applicant Name: Daniel R. Foster

Applicant Age: 37 years old

Annual Household Income: $72,400

Requested Loan Amount: $18,000 USD

Employment Status: Employed Full-Time (6 years with current employer)

Financial Context and Explanation (Unstructured Text)

I am applying for a loan to purchase a used mid-size sedan to replace my current vehicle, which has frequent mechanical issues and is no longer safe for long-distance commuting. My job requires me to travel between multiple office locations each week, so having reliable transportation is critical.

My financial history is generally positive—I maintain an average credit score of 705 and have consistently paid all previous debts on time. I currently have one active credit card and a small remaining balance on a previous personal loan, both of which are in good standing. My monthly expenses are stable and well within my income level.

I do want to disclose that I had two late payments last year due to temporary income disruption when I switched jobs. Since then, my finances have stabilized and I have built a more reliable budgeting system. I am confident in my ability to meet the repayment schedule without issue.

Certification

Signature: D. R. Foster
Date: November 26, 2025'''
}
res = integrations.run_feature_extraction(sample, mock=True)
print('--- PARSED ---')
print(json.dumps(res.get('parsed'), indent=2, ensure_ascii=False))
print('--- FEATURES ---')
print(json.dumps(res.get('features'), indent=2, ensure_ascii=False))
from backend.integrations import expand_parsed_to_fields, predict
fields = expand_parsed_to_fields(res.get('parsed'))
merged = {**res.get('features',{}), **fields}
print('--- MERGED FIELDS ---')
print(json.dumps(merged, indent=2, ensure_ascii=False))
print('--- PREDICT ---')
print(predict(merged))

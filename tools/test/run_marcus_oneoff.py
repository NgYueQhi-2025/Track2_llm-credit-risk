import sys, os
_THIS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from backend import integrations
import json

sample = {
  'id': 'marcus-1',
  'name': 'Marcus J. Delgado',
  'age': 41,
  'income': 64000,
  'requested_loan': 25000,
  'text_notes': '''Personal Information and Loan Request

Applicant Name: Marcus J. Delgado

Applicant Age: 41 years old

Annual Household Income: $64,000

Requested Loan Amount: $25,000 USD

Employment Status: Employed Full-Time (Manufacturing Technician)

Financial Context and Explanation (Unstructured Text)

I am requesting a loan for debt consolidation. Over the past two years, my family experienced several unexpected expenses, including home repairs and my father’s medical bills. These events caused me to rely heavily on multiple credit cards, resulting in higher-than-expected interest payments. I currently hold four active credit cards and two of them are near their limit. My credit score is 647—not ideal, but gradually improving. I have never defaulted on a loan, although I did have a few late payments earlier this year during a period of reduced overtime hours at work. This consolidation loan will allow me to simplify my payments and reduce interest charges, putting me on a more stable financial path. With my income returning to normal and my budgeting improved, I am fully prepared to commit to the repayment plan.

Certification

Signature: M. J. Delgado
Date: November 26, 2025'''
}

print('Running Marcus sample through run_feature_extraction (mock=True)')
res = integrations.run_feature_extraction(sample, mock=True)
print('\n--- PARSED ---')
print(json.dumps(res.get('parsed'), indent=2, ensure_ascii=False))
print('\n--- FEATURES ---')
print(json.dumps(res.get('features'), indent=2, ensure_ascii=False))

from backend.integrations import expand_parsed_to_fields, predict
fields = expand_parsed_to_fields(res.get('parsed'))
merged = {**res.get('features',{}), **fields}
print('\n--- MERGED FIELDS ---')
print(json.dumps(merged, indent=2, ensure_ascii=False))
print('\n--- PREDICT ---')
print(predict(merged))

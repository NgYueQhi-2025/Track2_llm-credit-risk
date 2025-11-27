from backend import integrations

sample = {
    'id': 'daniel-1',
    'name': 'Daniel R. Foster',
    'age': 37,
    'income': 72400,
    'requested_loan': 18000,
    'text_notes': """
I am applying for a loan to purchase a used mid-size sedan to replace my current vehicle, which has frequent mechanical issues and is no longer safe for long-distance commuting. My job requires me to travel between multiple office locations each week, so having reliable transportation is critical.

My financial history is generally positive—I maintain an average credit score of 705 and have consistently paid all previous debts on time. I currently have one active credit card and a small remaining balance on a previous personal loan, both of which are in good standing. My monthly expenses are stable and well within my income level.

I do want to disclose that I had two late payments last year due to temporary income disruption when I switched jobs. Since then, my finances have stabilized and I have built a more reliable budgeting system. I am confident in my ability to meet the repayment schedule without issue.
"""
}

res = integrations.run_feature_extraction(sample, mock=False)
print('\n=== Parsed ===')
print(res.get('parsed'))
print('\n=== Features ===')
print(res.get('features'))
from backend.integrations import expand_parsed_to_fields, predict
fields = expand_parsed_to_fields(res.get('parsed'))
print('\n=== Expanded fields ===')
print(fields)
print('\n=== Prediction ===')
print(predict({**res.get('features', {}), **fields}))

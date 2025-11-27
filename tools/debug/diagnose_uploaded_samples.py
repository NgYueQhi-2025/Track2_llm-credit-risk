import os
import sys
import re

# Ensure project root is on sys.path so `from backend import integrations` works
root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
if root not in sys.path:
    sys.path.insert(0, root)

from backend import integrations


def parse_fields_from_text_local(text: str) -> dict:
    """Lightweight parser for diagnostics (avoids importing backend.app which runs Streamlit)."""
    out = {}
    try:
        m = re.search(r"Applicant Name:\s*(.+)", text)
        if m:
            out['name'] = m.group(1).strip()
        m = re.search(r"Applicant Age:\s*(\d{1,3})", text)
        if m:
            out['age'] = int(m.group(1))
        m = re.search(r"Annual Household Income:\s*\$?([0-9,]+)", text)
        if m:
            out['income'] = float(m.group(1).replace(',', ''))
        m = re.search(r"Requested Loan Amount:\s*\$?([0-9,]+)", text)
        if m:
            out['requested_loan'] = float(m.group(1).replace(',', ''))
    except Exception:
        pass
    return out

SAMPLE_MULTI = """
■ Loan Application Sample 1
Personal Information and Loan Request
Applicant Name: Applicant 1 Example
Applicant Age: 26 years old
Annual Household Income: $51500
Requested Loan Amount: $12000 USD
Employment Status: Employed Full-Time
Financial Context and Explanation (Unstructured Text)
I am applying for this loan to help manage several financial obligations that have built up over the past
year. Unexpected expenses and temporary income changes have increased my reliance on credit,
leading to higher monthly payments.
This loan will allow me to consolidate debt, reduce interest, and better manage my finances moving
forward. I have maintained a consistent payment history and am committed to repaying the loan on
schedule.
Certification
Signature: Applicant 1
Date: November 27, 2025

■ Loan Application Sample 2
Personal Information and Loan Request
Applicant Name: Applicant 2 Example
Applicant Age: 27 years old
Annual Household Income: $53000
Requested Loan Amount: $14000 USD
Employment Status: Employed Full-Time
Financial Context and Explanation (Unstructured Text)
I am applying for this loan to help manage several financial obligations that have built up over the past
year. Unexpected expenses and temporary income changes have increased my reliance on credit,
leading to higher monthly payments.
This loan will allow me to consolidate debt, reduce interest, and better manage my finances moving
forward. I have maintained a consistent payment history and am committed to repaying the loan on
schedule.
Certification
Signature: Applicant 2
Date: November 27, 2025

■ Loan Application Sample 3
Personal Information and Loan Request
Applicant Name: Applicant 3 Example
Applicant Age: 28 years old
Annual Household Income: $54500
Requested Loan Amount: $16000 USD
Employment Status: Employed Full-Time
Financial Context and Explanation (Unstructured Text)
I am applying for this loan to help manage several financial obligations that have built up over the past
year. Unexpected expenses and temporary income changes have increased my reliance on credit,
leading to higher monthly payments.
This loan will allow me to consolidate debt, reduce interest, and better manage my finances moving
forward. I have maintained a consistent payment history and am committed to repaying the loan on
schedule.
Certification
Signature: Applicant 3
Date: November 27, 2025

■ Loan Application Sample 4
Personal Information and Loan Request
Applicant Name: Applicant 4 Example
Applicant Age: 29 years old
Annual Household Income: $56000
Requested Loan Amount: $18000 USD
Employment Status: Employed Full-Time
Financial Context and Explanation (Unstructured Text)
I am applying for this loan to help manage several financial obligations that have built up over the past
year. Unexpected expenses and temporary income changes have increased my reliance on credit,
leading to higher monthly payments.
This loan will allow me to consolidate debt, reduce interest, and better manage my finances moving
forward. I have maintained a consistent payment history and am committed to repaying the loan on
schedule.
Certification
Signature: Applicant 4
Date: November 27, 2025

■ Loan Application Sample 5
Personal Information and Loan Request
Applicant Name: Applicant 5 Example
Applicant Age: 30 years old
Annual Household Income: $57500
Requested Loan Amount: $20000 USD
Employment Status: Employed Full-Time
Financial Context and Explanation (Unstructured Text)
I am applying for this loan to help manage several financial obligations that have built up over the past
year. Unexpected expenses and temporary income changes have increased my reliance on credit,
leading to higher monthly payments.
This loan will allow me to consolidate debt, reduce interest, and better manage my finances moving
forward. I have maintained a consistent payment history and am committed to repaying the loan on
schedule.
Certification
Signature: Applicant 5
Date: November 27, 2025

■ Loan Application Sample 6
Personal Information and Loan Request
Applicant Name: Applicant 6 Example
Applicant Age: 31 years old
Annual Household Income: $59000
Requested Loan Amount: $22000 USD
Employment Status: Employed Full-Time
Financial Context and Explanation (Unstructured Text)
I am applying for this loan to help manage several financial obligations that have built up over the past
year. Unexpected expenses and temporary income changes have increased my reliance on credit,
leading to higher monthly payments.
This loan will allow me to consolidate debt, reduce interest, and better manage my finances moving
forward. I have maintained a consistent payment history and am committed to repaying the loan on
schedule.
Certification
Signature: Applicant 6
Date: November 27, 2025

■ Loan Application Sample 7
Personal Information and Loan Request
Applicant Name: Applicant 7 Example
Applicant Age: 32 years old
Annual Household Income: $60500
Requested Loan Amount: $24000 USD
Employment Status: Employed Full-Time
Financial Context and Explanation (Unstructured Text)
I am applying for this loan to help manage several financial obligations that have built up over the past
year. Unexpected expenses and temporary income changes have increased my reliance on credit,
leading to higher monthly payments.
This loan will allow me to consolidate debt, reduce interest, and better manage my finances moving
forward. I have maintained a consistent payment history and am committed to repaying the loan on
schedule.
Certification
Signature: Applicant 7
Date: November 27, 2025

■ Loan Application Sample 8
Personal Information and Loan Request
Applicant Name: Applicant 8 Example
Applicant Age: 33 years old
Annual Household Income: $62000
Requested Loan Amount: $26000 USD
Employment Status: Employed Full-Time
Financial Context and Explanation (Unstructured Text)
I am applying for this loan to help manage several financial obligations that have built up over the past
year. Unexpected expenses and temporary income changes have increased my reliance on credit,
leading to higher monthly payments.
This loan will allow me to consolidate debt, reduce interest, and better manage my finances moving
forward. I have maintained a consistent payment history and am committed to repaying the loan on
schedule.
Certification
Signature: Applicant 8
Date: November 27, 2025

■ Loan Application Sample 9
Personal Information and Loan Request
Applicant Name: Applicant 9 Example
Applicant Age: 34 years old
Annual Household Income: $63500
Requested Loan Amount: $28000 USD
Employment Status: Employed Full-Time
Financial Context and Explanation (Unstructured Text)
I am applying for this loan to help manage several financial obligations that have built up over the past
year. Unexpected expenses and temporary income changes have increased my reliance on credit,
leading to higher monthly payments.
This loan will allow me to consolidate debt, reduce interest, and better manage my finances moving
forward. I have maintained a consistent payment history and am committed to repaying the loan on
schedule.
Certification
Signature: Applicant 9
Date: November 27, 2025

■ Loan Application Sample 10
Personal Information and Loan Request
Applicant Name: Applicant 10 Example
Applicant Age: 35 years old
Annual Household Income: $65000
Requested Loan Amount: $30000 USD
Employment Status: Employed Full-Time
Financial Context and Explanation (Unstructured Text)
I am applying for this loan to help manage several financial obligations that have built up over the past
year. Unexpected expenses and temporary income changes have increased my reliance on credit,
leading to higher monthly payments.
This loan will allow me to consolidate debt, reduce interest, and better manage my finances moving
forward. I have maintained a consistent payment history and am committed to repaying the loan on
schedule.
Certification
Signature: Applicant 10
Date: November 27, 2025
"""


def run():
    chunks = integrations.split_text_into_applications(SAMPLE_MULTI)
    print(f"Detected {len(chunks)} chunks")
    for i, chunk in enumerate(chunks, start=1):
        print("\n" + "="*60)
        print(f"Applicant {i} chunk start:\n{chunk[:300]}...\n")
        parsed_fields = parse_fields_from_text_local(chunk)
        print("Parsed fields:", parsed_fields)
        row = {
            'id': i,
            'name': parsed_fields.get('name') or f'Applicant {i}',
            'age': parsed_fields.get('age'),
            'income': parsed_fields.get('income'),
            'requested_loan': parsed_fields.get('requested_loan'),
            'text_notes': chunk,
        }
        res = integrations.run_feature_extraction(row, mock=False)
        parsed = res.get('parsed')
        features = res.get('features')
        print("Parsed (LLM/fallback):", parsed)
        print("Features:", features)
        pred = integrations.predict(features)
        print("Prediction:", pred)

if __name__ == '__main__':
    run()

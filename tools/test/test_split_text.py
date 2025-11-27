from backend import integrations
from backend import app


SAMPLE_MULTI = """
■ Loan Application Sample 1 Personal Information and Loan Request Applicant Name: Applicant 1 Example Applicant Age: 26 years old Annual Household Income: $51500 Requested Loan Amount: $12000 USD Employment Status: Employed Full-Time Financial Context and Explanation (Unstructured Text) I am applying for this loan to help manage several financial obligations that have built up over the past year. Certification Signature: Applicant 1 Date: November 27, 2025

■ Loan Application Sample 2 Personal Information and Loan Request Applicant Name: Applicant 2 Example Applicant Age: 27 years old Annual Household Income: $53000 Requested Loan Amount: $14000 USD Employment Status: Employed Full-Time Financial Context and Explanation (Unstructured Text) Certification Signature: Applicant 2 Date: November 27, 2025

■ Loan Application Sample 3 Personal Information and Loan Request Applicant Name: Applicant 3 Example Applicant Age: 28 years old Annual Household Income: $54500 Requested Loan Amount: $16000 USD Employment Status: Employed Full-Time Financial Context and Explanation (Unstructured Text) Certification Signature: Applicant 3 Date: November 27, 2025

■ Loan Application Sample 4 Personal Information and Loan Request Applicant Name: Applicant 4 Example Applicant Age: 29 years old Annual Household Income: $56000 Requested Loan Amount: $18000 USD Employment Status: Employed Full-Time Financial Context and Explanation (Unstructured Text) Certification Signature: Applicant 4 Date: November 27, 2025

■ Loan Application Sample 5 Personal Information and Loan Request Applicant Name: Applicant 5 Example Applicant Age: 30 years old Annual Household Income: $57500 Requested Loan Amount: $20000 USD Employment Status: Employed Full-Time Financial Context and Explanation (Unstructured Text) Certification Signature: Applicant 5 Date: November 27, 2025

■ Loan Application Sample 6 Personal Information and Loan Request Applicant Name: Applicant 6 Example Applicant Age: 31 years old Annual Household Income: $59000 Requested Loan Amount: $22000 USD Employment Status: Employed Full-Time Financial Context and Explanation (Unstructured Text) Certification Signature: Applicant 6 Date: November 27, 2025

■ Loan Application Sample 7 Personal Information and Loan Request Applicant Name: Applicant 7 Example Applicant Age: 32 years old Annual Household Income: $60500 Requested Loan Amount: $24000 USD Employment Status: Employed Full-Time Financial Context and Explanation (Unstructured Text) Certification Signature: Applicant 7 Date: November 27, 2025

■ Loan Application Sample 8 Personal Information and Loan Request Applicant Name: Applicant 8 Example Applicant Age: 33 years old Annual Household Income: $62000 Requested Loan Amount: $26000 USD Employment Status: Employed Full-Time Financial Context and Explanation (Unstructured Text) Certification Signature: Applicant 8 Date: November 27, 2025

■ Loan Application Sample 9 Personal Information and Loan Request Applicant Name: Applicant 9 Example Applicant Age: 34 years old Annual Household Income: $63500 Requested Loan Amount: $28000 USD Employment Status: Employed Full-Time Financial Context and Explanation (Unstructured Text) Certification Signature: Applicant 9 Date: November 27, 2025

■ Loan Application Sample 10 Personal Information and Loan Request Applicant Name: Applicant 10 Example Applicant Age: 35 years old Annual Household Income: $65000 Requested Loan Amount: $30000 USD Employment Status: Employed Full-Time Financial Context and Explanation (Unstructured Text) Certification Signature: Applicant 10 Date: November 27, 2025
"""


def test_split_into_ten_chunks_and_parse_names():
    chunks = integrations.split_text_into_applications(SAMPLE_MULTI)
    # Expect 10 chunks for the 10 samples in the sample text
    assert isinstance(chunks, list)
    assert len(chunks) == 10, f"expected 10 chunks, got {len(chunks)}"

    # Ensure each chunk parses to the expected applicant name
    names = []
    for c in chunks:
        parsed = app.parse_fields_from_text(c)
        names.append(parsed.get('name'))

    # Check each applicant name is present and corresponds to Applicant 1..10
    for i in range(1, 11):
        expected = f"Applicant {i}"
        assert any(expected in (n or '') for n in names), f"{expected} not found in parsed names: {names}"

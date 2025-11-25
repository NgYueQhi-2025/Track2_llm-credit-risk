"""
Generate a large synthetic applicants CSV for testing the Streamlit app.
This script is deterministic (fixed seed) and creates realistic-looking sentences
covering different risk signals: late payments, cashflow issues, contradictions,
optimistic growth language, medical bills, bankruptcy, etc.

Usage (from repo root):
    py -3 scripts\generate_large_sample_csv.py --rows 200 --out sample_applicants_large.csv

If you omit --rows it defaults to 200.
"""
import csv
import random
import argparse
from datetime import date, timedelta

SEED = 42
random.seed(SEED)

risk_phrases = [
    "late payments",
    "unpaid invoices",
    "cashflow tight",
    "medical bills",
    "defaulted loan",
    "bankruptcy",
    "contradictory statements",
    "urgent plea",
    "no documentation",
    "recent promotion",
    "stable employment",
    "strong assets",
]

base_templates = [
    "I am applying to refinance after a temporary hardship; I had {rp} but have since returned to steady work and increased savings.",
    "Small business owner seeking expansion. Last quarter was {rp} but a new contract should improve revenue.",
    "Short application. Mentions 'self-employed' and provides little documentation; tone is vague.",
    "Stable employment for many years. Conservative spending and strong repayment history; a single late payment five years ago.",
    "Applying for personal loan to cover unexpected {rp}; enrolled in a repayment plan with the hospital.",
    "Requesting business credit; suppliers report {rp} and tone is defensive with repeated urgency.",
    "Education loan consolidation; emphasizes repayment intent and steady part-time work.",
    "Application contains inconsistent statements about employment and income, with {rp} mentioned.",
    "High income and low leverage; provides financial statements and conservative tone.",
    "Mentions previous bankruptcy and recent recovery; claims improved cashflow but offers limited proof.",
]

first_names = [
    "Alex","Taylor","Jordan","Morgan","Casey","Riley","Jamie","Avery","Parker","Quinn",
    "Robin","Skyler","Reese","Cameron","Drew","Hayden","Rowan","Emerson","Finley","Harper",
]
last_names = [
    "Smith","Johnson","Garcia","Brown","Davis","Miller","Wilson","Moore","Taylor","Anderson",
    "Thomas","Jackson","White","Harris","Martin","Thompson","Martinez","Robinson","Clark","Rodriguez",
]

def make_text(i):
    # Choose template and fill with a random risk phrase when needed
    t = random.choice(base_templates)
    rp = random.choice(risk_phrases)
    text = t.replace("{rp}", rp)
    # add a second sentence sometimes
    if random.random() < 0.45:
        extra = random.choice([
            "I expect the situation to normalize in the next quarter.",
            "I have attached bank statements and invoices to show recent cashflow.",
            "I am optimistic about upcoming contracts and a promotion.",
            "This is an urgent request due to unexpected circumstances.",
            "I have consolidated my debts and started a repayment plan.",
        ])
        text = text + " " + extra
    # occasionally insert quoted emotional phrase
    if random.random() < 0.15:
        text += " The application contains phrases like 'I can't pay' and 'urgent help needed'."
    return text


def make_row(i, start_date):
    fname = random.choice(first_names)
    lname = random.choice(last_names)
    name = f"{fname} {lname}"
    age = 22 + (i % 45)
    income = 25000 + (random.randint(0, 120) * 1000)  # 25k .. 145k
    credit_score = 500 + (random.randint(0, 350))
    text_notes = make_text(i)
    group = chr(ord('A') + (i % 4))
    # rough default label heuristic for synthetic truth (not used by mock model)
    default_label = 1 if (credit_score < 620 or any(k in text_notes for k in ["bankruptcy","defaulted","can't pay","unpaid"])) else 0
    app_date = start_date + timedelta(days=(i % 365))
    return {
        "id": 1000 + i,
        "name": name,
        "age": age,
        "income": income,
        "credit_score": credit_score,
        "text_notes": text_notes,
        "group": group,
        "default_label": default_label,
        "application_date": app_date.isoformat(),
    }


def generate(rows=200, out="sample_applicants_large.csv"):
    start = date(2024, 1, 1)
    with open(out, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["id","name","age","income","credit_score","text_notes","group","default_label","application_date"])
        writer.writeheader()
        for i in range(1, rows + 1):
            row = make_row(i, start)
            writer.writerow(row)
    print(f"Wrote {rows} rows to {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=200, help='Number of rows to generate')
    parser.add_argument('--out', type=str, default='sample_applicants_large.csv', help='Output CSV filename')
    args = parser.parse_args()
    generate(rows=args.rows, out=args.out)

"""Create demo artifacts: toy model.pkl, features.csv, and a demo CSV.

Run this script from the repo root (PowerShell):

    python backend\artifacts\setup_demo.py

It will create:
- backend/artifacts/model.pkl  (sklearn LogisticRegression trained on tiny synthetic data)
- backend/artifacts/features.csv (precomputed features matching demo applicants)
- data/demo.csv (applicant CSV used by the mock flow)

"""
import os
import csv
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
DATA = ROOT / "data"
ART.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

# 1) create demo CSV
demo_rows = [
    {"id": 101, "name": "Alice A.", "age": 31, "income": 52000},
    {"id": 102, "name": "Bob A.", "age": 45, "income": 72000},
    {"id": 103, "name": "Carol A.", "age": 38, "income": 31000},
]
with open(DATA / "demo.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "name", "age", "income"])
    writer.writeheader()
    for r in demo_rows:
        writer.writerow(r)

# 2) create precomputed features.csv matching applicant ids
features = [
    {"applicant_id": 101, "sentiment_score": 0.05, "risky_phrase_count": 0, "contradiction_flag": 0, "credibility_score": 0.65},
    {"applicant_id": 102, "sentiment_score": -0.2, "risky_phrase_count": 2, "contradiction_flag": 1, "credibility_score": 0.15},
    {"applicant_id": 103, "sentiment_score": 0.1, "risky_phrase_count": 1, "contradiction_flag": 0, "credibility_score": 0.5},
]
with open(ART / "features.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["applicant_id", "sentiment_score", "risky_phrase_count", "contradiction_flag", "credibility_score"])
    writer.writeheader()
    for r in features:
        writer.writerow(r)

# 3) train a tiny logistic regression model on synthetic features
# features vector order: sentiment_score, risky_phrase_count, contradiction_flag, credibility_score
X = np.array([[r["sentiment_score"], r["risky_phrase_count"], r["contradiction_flag"], r["credibility_score"]] for r in features])
# make simple labels: id 102 high risk, others low
y = np.array([0, 1, 0])
model = LogisticRegression()
model.fit(X, y)
joblib.dump(model, ART / "model.pkl")

print("Created demo artifacts:")
print(" -", ART / "model.pkl")
print(" -", ART / "features.csv")
print(" -", DATA / "demo.csv")
print("Run `streamlit run backend/app.py` and enable 'Mock mode' to use precomputed artifacts.")

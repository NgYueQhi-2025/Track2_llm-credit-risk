# ...existing code...
import os
import json
import csv
import joblib

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "demo_run")
OUT_DIR = os.path.abspath(OUT_DIR)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    features = [
        {"id": 1, "name": "Alice", "income": 50000, "credit_score": 700, "risky_phrase_count": 1, "summary": "used risky phrase", "sim_group": "A"},
        {"id": 2, "name": "Bob",   "income": 45000, "credit_score": 680, "risky_phrase_count": 0, "summary": "clean summary",        "sim_group": "B"},
    ]
    with open(os.path.join(OUT_DIR, "features.csv"), "w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(features[0].keys()))
        writer.writeheader()
        writer.writerows(features)

    scored = [
        {"id": 1, "name": "Alice", "risk_score": 0.75, "sim_group": "A"},
        {"id": 2, "name": "Bob",   "risk_score": 0.30, "sim_group": "B"},
    ]
    with open(os.path.join(OUT_DIR, "scored.csv"), "w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(scored[0].keys()))
        writer.writeheader()
        writer.writerows(scored)

    llm = {"1": {"summary": "used risky phrase", "risky_phrases": ["used risky phrase"]}}
    with open(os.path.join(OUT_DIR, "llm_outputs.json"), "w", encoding="utf8") as f:
        json.dump(llm, f, indent=2, ensure_ascii=False)

    joblib.dump({"dummy_model": True}, os.path.join(OUT_DIR, "model.pkl"))

    print("Placeholders created at", OUT_DIR)

if __name__ == "__main__":
    main()
# ...existing code...
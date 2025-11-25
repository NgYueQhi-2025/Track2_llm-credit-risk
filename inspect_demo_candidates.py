# ...existing code...
import os
import sys
import json

# try pandas, otherwise fall back to csv reader
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None

OUT_DIR = os.path.join("artifacts", "demo_run")
SCORED = os.path.join(OUT_DIR, "scored.csv")
FEATURES = os.path.join(OUT_DIR, "features.csv")
LLM = os.path.join(OUT_DIR, "llm_outputs.json")
OUT_SUMMARY = os.path.join(OUT_DIR, "inspect_summary.json")


def read_csv(path):
    if pd:
        return pd.read_csv(path)
    import csv
    with open(path, newline="", encoding="utf8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def main():
    if not os.path.isdir(OUT_DIR):
        print(f"Missing folder: {OUT_DIR} — run generator or create placeholders.", file=sys.stderr)
        sys.exit(1)

    if not (os.path.exists(SCORED) and os.path.exists(FEATURES)):
        print("Missing scored.csv or features.csv in artifacts/demo_run", file=sys.stderr)
        sys.exit(1)

    scored = read_csv(SCORED)
    features = read_csv(FEATURES)

    try:
        with open(LLM, "r", encoding="utf8") as f:
            llm = json.load(f)
    except Exception:
        llm = {}

    report = []
    if pd and hasattr(scored, "merge"):
        df = scored.merge(features, on=["id", "name"], how="left")
        if "risky_phrase_count" not in df.columns:
            df["risky_phrase_count"] = 0
        for _, row in df.iterrows():
            risk = float(row.get("risk_score", 0) or 0)
            risky_count = int(row.get("risky_phrase_count", 0) or 0)
            if risk > 0.5 or risky_count > 0:
                rid = str(int(row.id))
                report.append({
                    "id": int(row.id),
                    "name": row.name,
                    "risk_score": risk,
                    "risky_phrase_count": risky_count,
                    "summary": llm.get(rid, {}).get("summary", "")
                })
    else:
        # lists of dicts fallback
        feats_index = {(int(f["id"]), f.get("name")): f for f in features}
        for r in scored:
            rid = int(r.get("id"))
            name = r.get("name")
            risk = float(r.get("risk_score", 0) or 0)
            feat = feats_index.get((rid, name), {})
            risky_count = int(feat.get("risky_phrase_count", 0) or 0)
            if risk > 0.5 or risky_count > 0:
                sid = str(rid)
                report.append({
                    "id": rid,
                    "name": name,
                    "risk_score": risk,
                    "risky_phrase_count": risky_count,
                    "summary": llm.get(sid, {}).get("summary", "")
                })

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_SUMMARY, "w", encoding="utf8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("Wrote inspect summary to", OUT_SUMMARY)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
# ...existing code...
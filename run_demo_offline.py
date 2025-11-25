# ...existing code...
import os
import sys
import json
import csv
import joblib
import traceback
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(ROOT, "artifacts", "demo_run")
LOG_PATH = os.path.join(ROOT, "run_output.txt")


def write_log(msg: str):
    ts = datetime.now().isoformat(timespec="seconds")
    line = f"[{ts}] {msg}\n"
    with open(LOG_PATH, "a", encoding="utf8") as f:
        f.write(line)


def create_placeholders():
    os.makedirs(OUT_DIR, exist_ok=True)
    features = [
        {"id": 1, "name": "Alice", "income": 50000, "credit_score": 700, "risky_phrase_count": 1, "summary": "used risky phrase", "sim_group": "A"},
        {"id": 2, "name": "Bob",   "income": 45000, "credit_score": 680, "risky_phrase_count": 0, "summary": "clean summary",        "sim_group": "B"},
    ]
    with open(os.path.join(OUT_DIR, "features.csv"), "w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=features[0].keys())
        writer.writeheader()
        writer.writerows(features)

    scored = [
        {"id": 1, "name": "Alice", "risk_score": 0.75, "sim_group": "A"},
        {"id": 2, "name": "Bob",   "risk_score": 0.30, "sim_group": "B"},
    ]
    with open(os.path.join(OUT_DIR, "scored.csv"), "w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=scored[0].keys())
        writer.writeheader()
        writer.writerows(scored)

    json.dump({"1": {"summary": "used risky phrase", "risky_phrases": ["used risky phrase"]}}, open(os.path.join(OUT_DIR, "llm_outputs.json"), "w"))
    joblib.dump({"dummy": "model"}, os.path.join(OUT_DIR, "model.pkl"))
    write_log(f"Placeholders created at {OUT_DIR}")


def try_run_real_generator():
    """
    Attempt to import and run the project's real generator pipeline.
    Any import or runtime error will be logged and returned as False.
    """
    try:
        # these imports are best-effort; adjust names if your project differs
        from backend.generate_synthetic import generate_synthetic  # type: ignore
        from backend.feature_extraction import run_feature_extraction  # type: ignore
        from backend.backend.app import run_pipeline  # type: ignore
        write_log("Imported pipeline modules successfully.")
    except Exception:
        write_log("Failed to import pipeline modules:\n" + traceback.format_exc())
        return False

    try:
        write_log("Starting real generator...")
        # call functions if they accept output path; adapt if API differs
        try:
            generate_synthetic(OUT_DIR)
            write_log("generate_synthetic completed.")
        except TypeError:
            # fallback if function signature differs
            generate_synthetic()
            write_log("generate_synthetic completed (no args).")

        try:
            run_feature_extraction(OUT_DIR)
            write_log("run_feature_extraction completed.")
        except TypeError:
            run_feature_extraction()
            write_log("run_feature_extraction completed (no args).")

        try:
            run_pipeline(OUT_DIR)
            write_log("run_pipeline completed.")
        except TypeError:
            run_pipeline()
            write_log("run_pipeline completed (no args).")

        return True
    except Exception:
        write_log("Generator run failed:\n" + traceback.format_exc())
        return False


def main():
    # start fresh log
    with open(LOG_PATH, "w", encoding="utf8") as f:
        f.write(f"run_demo_offline.log started at {datetime.now().isoformat()}\n")

    write_log("Script started.")

    ok = try_run_real_generator()
    if ok:
        write_log(f"Wrote artifacts to {OUT_DIR} using real generator.")
        print("Wrote artifacts to", OUT_DIR)
        return

    write_log("Falling back to creating placeholders.")
    create_placeholders()
    print("Created placeholder artifacts at", OUT_DIR)
    write_log("Finished (placeholders).")


if __name__ == "__main__":
    main()
# ...existing code...
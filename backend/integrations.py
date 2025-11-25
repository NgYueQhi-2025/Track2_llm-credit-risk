import csv
import json
import os
import time
from typing import Any, Dict, Optional

import importlib.util
import joblib


# Attempt to import the LLM handler from the `llms` package; if the
# handler file doesn't have a standard .py extension or the package
# isn't importable, fall back to loading it by file path so the demo
# still runs.
try:
    from llms.backend import llm_handler
except Exception:
    llm_path = os.path.join(os.path.dirname(__file__), "..", "llms", "backend", "llm_handler")
    llm_path = os.path.normpath(llm_path)
    if os.path.exists(llm_path):
            # Try to load via importlib; if that fails (e.g. file has no .py
            # extension and spec.loader is None), fall back to executing the
            # source text into a new module namespace.
            spec = importlib.util.spec_from_file_location("llm_handler", llm_path)
            if spec is not None and spec.loader is not None:
                llm_handler = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(llm_handler)
            else:
                # Read source and exec into module
                import types

                llm_handler = types.ModuleType("llm_handler")
                # populate typical module attributes so code using __file__ or
                # __package__ behaves as expected
                llm_handler.__file__ = llm_path
                llm_handler.__package__ = "llms.backend"
                llm_handler.__name__ = "llms.backend.llm_handler"
                with open(llm_path, "r", encoding="utf-8") as f:
                    src = f.read()
                exec(compile(src, llm_path, "exec"), llm_handler.__dict__)
    else:
        # Last resort: raise the original import error for visibility
        raise


MODEL_PATHS = [
    os.path.join(os.path.dirname(__file__), "artifacts", "model.pkl"),
    os.path.join(os.path.dirname(__file__), "artifacts", "model.joblib"),
    os.path.join(os.path.dirname(__file__), "..", "model.pkl"),
]

ARTIFACT_FEATURES = os.path.join(os.path.dirname(__file__), "artifacts", "features.csv")


def _read_precomputed_features(applicant_id: str) -> Optional[Dict[str, Any]]:
    """Return a features dict for applicant_id if found in ARTIFACT_FEATURES; else None."""
    if not os.path.exists(ARTIFACT_FEATURES):
        return None
    try:
        with open(ARTIFACT_FEATURES, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if str(r.get("applicant_id")) == str(applicant_id):
                    # cast numeric fields
                    return {
                        "applicant_id": r.get("applicant_id"),
                        "sentiment_score": float(r.get("sentiment_score", 0.0)),
                        "risky_phrase_count": int(r.get("risky_phrase_count", 0)),
                        "contradiction_flag": int(r.get("contradiction_flag", 0)),
                        "credibility_score": float(r.get("credibility_score", 0.0)),
                    }
    except Exception:
        return None
    return None


def _safe_json_load(s: str) -> Any:
    if not isinstance(s, str):
        return s
    s = s.strip()
    # Try to extract a JSON object substring
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    # fallback direct load
    try:
        return json.loads(s)
    except Exception:
        return s


def run_feature_extraction(df_row: Dict[str, Any], mock: bool = True, max_retries: int = 3) -> Dict[str, Any]:
    """Call LLM to extract structured outputs for a single applicant row.

    Returns a dict containing parsed LLM outputs and derived numeric features.

    - df_row: mapping of applicant fields (e.g. id, name, age, income, free_text...)
    - mock: when True the `llm_handler` will return canned responses
    - max_retries: retries for transient failures
    """
    applicant_id = str(df_row.get("id", "unknown"))

    prompt_base = f"Applicant data: {json.dumps(df_row, ensure_ascii=False)}\nAnalyze for: summary, risky phrases, contradictions, sentiment."

    modes = ["summary", "extract_risky", "detect_contradictions", "sentiment"]
    parsed: Dict[str, Any] = {}

    for mode in modes:
        attempt = 0
        last_exc = None
        while attempt < max_retries:
            # If mock and a precomputed artifact exists, use it for speed and determinism
            if mock:
                pre = _read_precomputed_features(applicant_id)
                if pre is not None:
                    # create minimal parsed structures consistent with feature_extraction expectations
                    parsed = {
                        "summary": {"summary": "(precomputed)", "confidence": 0.5},
                        "extract_risky": {"risky_phrases": [], "count": pre.get("risky_phrase_count", 0)},
                        "detect_contradictions": {"contradictions": [], "flag": pre.get("contradiction_flag", 0)},
                        "sentiment": {"sentiment": "neutral", "score": pre.get("sentiment_score", 0.0)},
                    }
                    # we have all features already: return early
                    features = {
                        "applicant_id": applicant_id,
                        "sentiment_score": pre.get("sentiment_score", 0.0),
                        "risky_phrase_count": pre.get("risky_phrase_count", 0),
                        "contradiction_flag": pre.get("contradiction_flag", 0),
                        "credibility_score": pre.get("credibility_score", 0.0),
                    }
                    return {"parsed": parsed, "features": features}

            try:
                raw = llm_handler.call_llm(prompt_base, mode=mode, temperature=0.0, use_cache=True, mock=mock)
                parsed_val = _safe_json_load(raw)
                parsed[mode] = parsed_val
                break
            except Exception as e:
                last_exc = e
                attempt += 1
                wait = 0.5 * (2 ** attempt)
                time.sleep(wait)
        else:
            # all retries failed
            raise RuntimeError(f"LLM extraction failed for applicant {applicant_id} mode={mode}: {last_exc}")

    # Map parsed outputs to numeric features (same logic as feature_extraction mapper)
    # sentiment
    sent = parsed.get("sentiment", {})
    if isinstance(sent, dict):
        sentiment_score = float(sent.get("score", 0.0))
    else:
        sentiment_score = 0.0

    # risky
    risky = parsed.get("extract_risky", {})
    if isinstance(risky, dict):
        risky_count = int(risky.get("count", len(risky.get("risky_phrases", []))))
    else:
        risky_count = 0

    # contradictions
    contra = parsed.get("detect_contradictions", {})
    contradiction_flag = int(contra.get("flag", 0)) if isinstance(contra, dict) else 0

    # credibility_score heuristic
    summary = parsed.get("summary", {})
    conf = float(summary.get("confidence", 0.5)) if isinstance(summary, dict) else 0.5
    credibility_score = max(0.0, min(1.0, conf - 0.3 * contradiction_flag - 0.05 * risky_count))

    features = {
        "applicant_id": applicant_id,
        "sentiment_score": sentiment_score,
        "risky_phrase_count": risky_count,
        "contradiction_flag": contradiction_flag,
        "credibility_score": credibility_score,
    }

    return {"parsed": parsed, "features": features}


def predict(features: Dict[str, Any]) -> Dict[str, Any]:
    """Given a features mapping, return a predicted risk score and label.

    Attempts to load a trained model from known artifact locations; if not found,
    falls back to a deterministic heuristic scoring function.
    """
    model = None
    for p in MODEL_PATHS:
        if os.path.exists(p):
            try:
                model = joblib.load(p)
                break
            except Exception:
                model = None

    # Convert features to vector in expected order
    vec = [
        float(features.get("sentiment_score", 0.0)),
        float(features.get("risky_phrase_count", 0.0)),
        float(features.get("contradiction_flag", 0.0)),
        float(features.get("credibility_score", 0.0)),
    ]

    if model is not None:
        try:
            score = float(model.predict_proba([vec])[0][1])
        except Exception:
            # model present but interface unexpected; fallback
            score = None
    else:
        score = None

    if score is None:
        # simple deterministic heuristic: higher risky count and low credibility -> higher risk
        score = max(0.0, min(1.0, 0.4 * (vec[1] / (1 + vec[1])) + 0.35 * (1 - vec[3]) + 0.25 * vec[2]))

    label = "low" if score < 0.5 else "high"

    return {"score": float(score), "risk_label": label}

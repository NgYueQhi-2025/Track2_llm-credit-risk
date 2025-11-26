import csv
import json
import os
import time
from typing import Any, Dict, Optional, List

import importlib.util
import joblib
import inspect


# Attempt to import the LLM handler from the `llms` package; if the
# handler file doesn't have a standard .py extension or the package
# isn't importable, fall back to loading it by file path so the demo
# still runs.
try:
    # IMPORTANT: Your file structure needs to support this relative import.
    # e.g., if this file is in 'backend/' and llm_handler.py is in 'llms/backend/'
    from llms.backend import llm_handler
except Exception as original_exc:
    # If the standard import fails, try to load it manually
    llm_path = os.path.join(os.path.dirname(__file__), "..", "llms", "backend", "llm_handler.py")
    llm_path = os.path.normpath(llm_path)
    if os.path.exists(llm_path):
        spec = importlib.util.spec_from_file_location("llm_handler", llm_path)
        if spec is not None and spec.loader is not None:
            llm_handler = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(llm_handler)
            except Exception as load_exc:
                raise ImportError(f"Could not execute llm_handler module from {llm_path}: {load_exc}") from load_exc
        else:
            # Fallback for odd file types (very unlikely here)
            import types
            llm_handler = types.ModuleType("llm_handler")
            llm_handler.__file__ = llm_path
            llm_handler.__package__ = "llms.backend"
            llm_handler.__name__ = "llms.backend.llm_handler"
            with open(llm_path, "r", encoding="utf-8") as f:
                src = f.read()
            exec(compile(src, llm_path, "exec"), llm_handler.__dict__)
    else:
        # Last resort: raise the original import error for visibility
        raise ImportError(f"Could not import or load llm_handler from {llm_path} (original error: {original_exc})")

# Attempt to import a compatibility adapter that provides a canonical
# `call_llm(prompt, mode=..., temperature=..., use_cache=..., mock=...)`
# If available, we'll prefer that to reduce signature mismatches.
try:
    from llms.backend import call_llm_compat as llm_compat
    _CALL_LLM_FUNC = getattr(llm_compat, 'call_llm')
except Exception:
    _CALL_LLM_FUNC = None


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
                # Call the LLM handler in a backwards-compatible way. Prefer
                # the compatibility adapter if present; else use the raw
                # llm_handler.call_llm function.
                func = _CALL_LLM_FUNC if _CALL_LLM_FUNC is not None else getattr(llm_handler, 'call_llm')

                # Gather handler metadata and signature (used for logging/debugging)
                try:
                    handler_file = getattr(llm_handler, '__file__', str(llm_handler))
                except Exception:
                    handler_file = str(llm_handler)
                try:
                    sig = inspect.signature(func)
                    params = sig.parameters
                except Exception:
                    sig = None
                    params = {}
                print(f"DEBUG: Using llm_handler at {handler_file} with signature: {sig}")

                # Build kwargs only for parameters that actually exist on the function.
                kwargs = {}
                if 'mode' in params: # New check for handler that supports mode as kwarg
                    kwargs['mode'] = mode
                if 'temperature' in params:
                    kwargs['temperature'] = 0.0
                if 'use_cache' in params:
                    kwargs['use_cache'] = True
                
                # --- EDITED: Pass 'mock' state to the LLM call ---
                if 'mock' in params:
                    kwargs['mock'] = mock

                # Ensure the prompt explicitly states the mode for handlers
                # that only accept a prompt string.
                prompt_for_call = f"Mode: {mode}\n" + prompt_base

                raw = None
                # Call the function, dynamically handling arguments
                # We prioritize passing kwargs and try falling back to positional arguments
                try:
                    if kwargs:
                        # If handler accepts 'mode' as kwarg, we pass the base prompt.
                        # Otherwise, we pass the enriched prompt.
                        raw = func(prompt_base if 'mode' in kwargs else prompt_for_call, **kwargs)
                    else:
                        raw = func(prompt_for_call)
                except TypeError:
                    # Fallback on positional arguments if kwargs fail, using original logic
                    # Try prompt-only
                    try:
                        raw = func(prompt_for_call)
                    except Exception:
                        # Try common positional orders as last resorts
                        tried = False
                        try:
                            # Try positional: prompt, mode, temperature, use_cache, mock
                            raw = func(prompt_for_call, mode, 0.0, True, mock)
                            tried = True
                        except Exception:
                            pass
                        if not tried:
                            try:
                                # Try positional: prompt, mode
                                raw = func(prompt_for_call, mode)
                            except Exception:
                                # Final fallback: call with original prompt_base
                                raw = func(prompt_base)

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


def expand_parsed_to_fields(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a parsed LLM output into flat fields used by the app/UI.

    Returns a dict with keys:
      - summary: text or None
      - sentiment_score: float
      - risky_phrases: list
      - risky_phrase_count: int
      - contradiction_flag: int
      - credibility_score: float
    """
    out = {
        "summary": None,
        "sentiment_score": None,
        "risky_phrases": None,
        "risky_phrase_count": None,
        "contradiction_flag": None,
        "credibility_score": None,
    }
    if not isinstance(parsed, dict):
        return out

    # summary
    summary = parsed.get("summary")
    if isinstance(summary, dict):
        out["summary"] = summary.get("summary")
    elif isinstance(summary, str):
        out["summary"] = summary

    # sentiment
    sent = parsed.get("sentiment")
    if isinstance(sent, dict):
        try:
            out["sentiment_score"] = float(sent.get("score", None))
        except Exception:
            out["sentiment_score"] = None
    elif isinstance(sent, (int, float)):
        out["sentiment_score"] = float(sent)

    # risky phrases
    risky = parsed.get("extract_risky") or parsed.get("risky")
    if isinstance(risky, dict):
        rp = risky.get("risky_phrases") or risky.get("phrases") or []
        if isinstance(rp, str):
            # try to parse comma-separated string
            rp_list = [p.strip() for p in rp.split(",") if p.strip()]
        elif isinstance(rp, (list, tuple)):
            rp_list = list(rp)
        else:
            rp_list = []
        out["risky_phrases"] = rp_list
        try:
            out["risky_phrase_count"] = int(risky.get("count", len(rp_list)))
        except Exception:
            out["risky_phrase_count"] = len(rp_list)

    # contradictions
    contra = parsed.get("detect_contradictions") or parsed.get("contradictions")
    if isinstance(contra, dict):
        out["contradiction_flag"] = int(contra.get("flag", 0)) if contra.get("flag") is not None else (1 if contra.get("contradictions") else 0)

    # credibility score fallback (if present in parsed features)
    if out.get("credibility_score") is None:
        # try to derive from summary confidence if available
        if isinstance(summary, dict) and summary.get("confidence") is not None:
            try:
                out["credibility_score"] = max(0.0, min(1.0, float(summary.get("confidence", 0.0))))
            except Exception:
                out["credibility_score"] = None

    return out


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
    vec: List[float] = [
        float(features.get("sentiment_score", 0.0)),
        float(features.get("risky_phrase_count", 0.0)),
        float(features.get("contradiction_flag", 0.0)),
        float(features.get("credibility_score", 0.0)),
    ]

    if model is not None:
        try:
            # Assumes model.predict_proba returns [proba_class_0, proba_class_1]
            score = float(model.predict_proba([vec])[0][1])
        except Exception:
            # model present but interface unexpected; fallback
            score = None
    else:
        score = None

    if score is None:
        # simple deterministic heuristic: higher risky count and low credibility -> higher risk
        score = max(0.0, min(1.0, 0.55 * (vec[1] / (1 + vec[1])) + 0.25 * (1 - vec[3]) + 0.20 * vec[2]))

    # lower the threshold slightly so borderline cases surface as 'high' in demos
    label = "low" if score < 0.45 else "high"

    return {"score": float(score), "risk_label": label}

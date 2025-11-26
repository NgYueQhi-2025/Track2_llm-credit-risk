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
                # If the repository's llm_handler.py fails to execute (syntax
                # error or env-specific issue), fall back to a lightweight
                # deterministic stub so the demo app remains functional.
                import types
                stub = types.ModuleType("llms.backend.llm_handler")
                def _stub_call_llm(prompt, *args, **kwargs):
                    pm = str(prompt or "").lower()
                    if "mode: summary" in pm or pm.startswith('mode: summary'):
                        return {"summary": {"summary": "Applicant appears stable and able to repay.", "confidence": 0.6}}
                    if "sentiment" in pm:
                        return {"sentiment": {"score": 0.2}}
                    if "extract_risky" in pm or "risky" in pm:
                        return {"extract_risky": {"risky_phrases": ["opened new credit lines"], "count": 1}}
                    if "detect_contradictions" in pm or "contradictions" in pm:
                        return {"detect_contradictions": {"contradictions": [], "flag": 0}}
                    return {}
                stub.call_llm = _stub_call_llm
                llm_handler = stub
                # Log a helpful message to stdout so developers see the fallback
                print(f"WARNING: Loaded stub llm_handler because repository llm_handler at {llm_path} failed to execute: {load_exc}")
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
        # If file doesn't exist, provide a lightweight stub instead of
        # raising so the demo app can still run in mock mode.
        import types
        stub = types.ModuleType("llms.backend.llm_handler")
        def _stub_call_llm(prompt, *args, **kwargs):
            pm = str(prompt or "").lower()
            if "mode: summary" in pm or pm.startswith('mode: summary'):
                return {"summary": {"summary": "Applicant appears stable and able to repay.", "confidence": 0.6}}
            if "sentiment" in pm:
                return {"sentiment": {"score": 0.2}}
            if "extract_risky" in pm or "risky" in pm:
                return {"extract_risky": {"risky_phrases": ["opened new credit lines"], "count": 1}}
            if "detect_contradictions" in pm or "contradictions" in pm:
                return {"detect_contradictions": {"contradictions": [], "flag": 0}}
            return {}
        stub.call_llm = _stub_call_llm
        llm_handler = stub
        print(f"WARNING: llm_handler.py not found at {llm_path}; using fallback stub. Original error: {original_exc}")

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


def _normalize_llm_raw_output(mode: str, parsed_val: Any, raw: Any) -> Any:
    """Normalize non-JSON LLM outputs into the expected parsed dict shapes.

    If the LLM returned a plain string (e.g. free-text summary), convert it
    into a minimal dict structure that the rest of the pipeline expects.
    """
    if isinstance(parsed_val, dict):
        return parsed_val

    text = None
    if isinstance(parsed_val, str) and parsed_val.strip():
        text = parsed_val.strip()
    elif isinstance(raw, str) and raw.strip():
        text = raw.strip()

    if not text:
        return parsed_val

    # mode-specific heuristics
    low = text.lower()
    if mode == 'summary':
        # use the full text as summary, with a conservative confidence
        return {"summary": {"summary": text, "confidence": 0.5}}

    if mode == 'sentiment':
        # try to find a numeric score, else map sentiment words
        import re
        m = re.search(r"(-?\d+\.\d+|-?\d+)", text)
        if m:
            try:
                return {"sentiment": {"score": float(m.group(1))}}
            except Exception:
                pass
        if 'positive' in low or 'good' in low or 'stable' in low or 'favorable' in low:
            return {"sentiment": {"score": 0.2}}
        if 'negative' in low or 'concern' in low or 'risk' in low or 'problem' in low:
            return {"sentiment": {"score": -0.2}}
        return {"sentiment": {"score": 0.0}}

    if mode == 'extract_risky':
        # split by common separators into candidate risky phrases
        parts = [p.strip() for p in re.split(r"[\n,;\\-\\•]+", text) if p.strip()]
        # keep short candidate phrases (heuristic)
        phrases = [p for p in parts if 2 <= len(p) <= 200]
        return {"extract_risky": {"risky_phrases": phrases, "count": len(phrases)}}

    if mode == 'detect_contradictions':
        flag = 1 if 'contradict' in low or 'inconsistent' in low or 'contradiction' in low else 0
        return {"detect_contradictions": {"contradictions": [], "flag": flag}}

    return parsed_val


def run_feature_extraction(df_row: Dict[str, Any], mock: bool = True, max_retries: int = 3) -> Dict[str, Any]:
    """Call LLM to extract structured outputs for a single applicant row.

    Returns a dict containing parsed LLM outputs and derived numeric features.

    - df_row: mapping of applicant fields (e.g. id, name, age, income, free_text...)
    - mock: when True the `llm_handler` will return canned responses
    - max_retries: retries for transient failures
    """
    applicant_id = str(df_row.get("id", "unknown"))

    # DEBUG: surface incoming row content so we can confirm the text field is present
    try:
        sample_text = (df_row.get('text_notes') or df_row.get('text') or df_row.get('text_to_analyze') or '')[:1000]
    except Exception:
        sample_text = ''
    print(f"DEBUG run_feature_extraction START applicant={applicant_id} mock={mock} keys={list(df_row.keys())}")
    print(f"DEBUG sample_text (truncated, 1000 chars): {repr(sample_text)[:400]}")  # truncated so logs stay readable

    prompt_base = f"Applicant data: {json.dumps(df_row, ensure_ascii=False)}\nAnalyze for: summary, risky phrases, contradictions, sentiment."

    modes = ["summary", "extract_risky", "detect_contradictions", "sentiment"]
    parsed: Dict[str, Any] = {}

    # If precomputed artifact features exist for this applicant, prefer them
    # regardless of `mock` so demo runs reproduce precomputed behavior.
    pre = _read_precomputed_features(applicant_id)
    if pre is not None:
        parsed = {
            "summary": {"summary": "(precomputed)", "confidence": 0.5},
            "extract_risky": {"risky_phrases": [], "count": pre.get("risky_phrase_count", 0)},
            "detect_contradictions": {"contradictions": [], "flag": pre.get("contradiction_flag", 0)},
            "sentiment": {"sentiment": "neutral", "score": pre.get("sentiment_score", 0.0)},
        }
        features = {
            "applicant_id": applicant_id,
            "sentiment_score": pre.get("sentiment_score", 0.0),
            "risky_phrase_count": pre.get("risky_phrase_count", 0),
            "contradiction_flag": pre.get("contradiction_flag", 0),
            "credibility_score": pre.get("credibility_score", 0.0),
        }
        print(f"DEBUG: returning precomputed features for applicant={applicant_id}")
        return {"parsed": parsed, "features": features}

    for mode in modes:
        attempt = 0
        last_exc = None
        while attempt < max_retries:
            # If mock and a precomputed artifact exists, use it for speed and determinism
            if mock:
                pre = _read_precomputed_features(applicant_id)
                if pre is not None:
                    parsed = {
                        "summary": {"summary": "(precomputed)", "confidence": 0.5},
                        "extract_risky": {"risky_phrases": [], "count": pre.get("risky_phrase_count", 0)},
                        "detect_contradictions": {"contradictions": [], "flag": pre.get("contradiction_flag", 0)},
                        "sentiment": {"sentiment": "neutral", "score": pre.get("sentiment_score", 0.0)},
                    }
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

                # Ensure we forward the mock flag if the handler supports it.
                if 'mock' in params:
                    kwargs['mock'] = mock

                # Ensure the prompt explicitly states the mode for handlers
                # that only accept a prompt string.
                prompt_for_call = f"Mode: {mode}\n" + prompt_base

                # DEBUG: show what we're about to send to the handler (truncated)
                try:
                    dbg_prompt = prompt_for_call if len(prompt_for_call) < 1000 else prompt_for_call[:1000] + "..."
                    print(f"DEBUG calling llm_handler func name={getattr(func, '__name__', str(func))} mode={mode} mock={mock} prompt_excerpt={repr(dbg_prompt)[:400]}")
                except Exception:
                    pass

                raw = None
                # Call the function, dynamically handling arguments.
                # Prefer keyword-call if handler supports kwargs; otherwise try common positional signatures.
                try:
                    if kwargs:
                        # If handler accepts 'mode' as kwarg, pass the base prompt or prompt depending on handler expectations.
                        raw = func(prompt_base if 'mode' in kwargs else prompt_for_call, **kwargs)
                    else:
                        # Try several positional argument orders, including the common one with mock at the end.
                        try:
                            raw = func(prompt_for_call, mode, 0.0, True, mock)
                        except TypeError:
                            try:
                                raw = func(prompt_for_call, mode, 0.0, True)
                            except TypeError:
                                try:
                                    raw = func(prompt_for_call, mode)
                                except TypeError:
                                    raw = func(prompt_for_call)
                except TypeError as te:
                    # Last-resort positional attempts already tried; re-raise to handled by outer except
                    raise te

                parsed_val = _safe_json_load(raw)
                # Normalize plain-text or unexpected outputs into expected dict shapes
                parsed[mode] = _normalize_llm_raw_output(mode, parsed_val, raw)

                # DEBUG: print what we received (raw truncated + parsed snippet)
                try:
                    raw_dbg = (raw if raw is not None else '')[:1000]
                except Exception:
                    raw_dbg = "<unprintable>"
                print(f"DEBUG llm raw (truncated) mode={mode} applicant={applicant_id}: {repr(raw_dbg)[:400]}")
                try:
                    print(f"DEBUG parsed[mode] mode={mode} applicant={applicant_id}: {json.dumps(parsed[mode], ensure_ascii=False)[:800]}")
                except Exception:
                    print(f"DEBUG parsed[mode] (unserializable) mode={mode} applicant={applicant_id}")

                break
            except Exception as e:
                last_exc = e
                # If the error indicates a local LLM server is unreachable (connection refused),
                # fall back to a local heuristic driven by the user's text (not the canned mock).
                msg = str(e).lower()
                if any(x in msg for x in ("connection refused", "httpconnectionpool", "failed to establish a new connection", "max retries exceeded")):
                    print(f"WARNING: LLM provider unreachable ({e}); using local text-based fallback for mode={mode}")
                    try:
                        # Use the applicant text as the basis for the fallback heuristic
                        text_for_fallback = df_row.get('text_notes') or df_row.get('text') or ''
                        # Local heuristic per mode
                        if mode == 'extract_risky':
                            import re
                            rp_list = []
                            # Patterns tuned to common risky phrases; will only add matches present in text
                            patterns = [
                                r'open(ed)?\s+.*new\s+lines?\s+of\s+credit',
                                r'new\s+lines?\s+of\s+credit',
                                r'bankrupt',
                                r'default',
                                r'late\s+payment(s)?',
                                r'collection',
                                r'multiple\s+loans?',
                                r'overdraft',
                                r'missed\s+payment(s)?',
                            ]
                            for p in patterns:
                                m = re.search(p, text_for_fallback, flags=re.IGNORECASE)
                                if m:
                                    cand = m.group(0)
                                    if cand not in rp_list:
                                        rp_list.append(cand)
                            parsed[mode] = {'risky_phrases': rp_list, 'count': len(rp_list)}
                        elif mode == 'sentiment':
                            low = (text_for_fallback or '').lower()
                            # simple sentiment heuristic fallback
                            if re.search(r'\b(good|stable|positive|improve|improved|strong)\b', low):
                                parsed[mode] = {'sentiment': 'positive', 'score': 0.2}
                            elif re.search(r'\b(bad|negative|risk|problem|concern|poor|unstable|decline)\b', low):
                                parsed[mode] = {'sentiment': 'negative', 'score': -0.2}
                            else:
                                parsed[mode] = {'sentiment': 'neutral', 'score': 0.0}
                        elif mode == 'detect_contradictions':
                            # simple contradictions fallback: look for "contradict" keywords
                            import re
                            if re.search(r'contradict|inconsist|inconsistent|conflict', text_for_fallback, flags=re.IGNORECASE):
                                parsed[mode] = {'contradictions': ['detected_in_text'], 'flag': 1}
                            else:
                                parsed[mode] = {'contradictions': [], 'flag': 0}
                        else:
                            parsed[mode] = {}
                        # Print debug for fallback
                        try:
                            print(f"DEBUG fallback parsed[mode] mode={mode} for applicant={applicant_id}: {json.dumps(parsed[mode], ensure_ascii=False)[:800]}")
                        except Exception:
                            print(f"DEBUG fallback parsed[mode] (unserializable) mode={mode} for applicant={applicant_id}")
                        break
                    except Exception as e2:
                        last_exc = e2
                        attempt += 1
                        wait = 0.5 * (2 ** attempt)
                        time.sleep(wait)
                        continue
                attempt += 1
                wait = 0.5 * (2 ** attempt)
                time.sleep(wait)
        else:
            # all retries failed
            raise RuntimeError(f"LLM extraction failed for applicant {applicant_id} mode={mode}: {last_exc}")

    # Map parsed outputs to numeric features (same logic as feature_extraction mapper)
    # --- Augment parsed outputs with simple text-based fallbacks ---
    def _augment_parsed_with_text(parsed: Dict[str, Any], text: str) -> Dict[str, Any]:
        """Ensure parsed contains risky/contradiction signals by scanning raw text as a fallback."""
        try:
            import re
            if not isinstance(parsed, dict):
                parsed = {}
            # detect contradictions keywords
            det = parsed.get('detect_contradictions') or {}
            if not det or (isinstance(det, dict) and det.get('flag', 0) == 0):
                if isinstance(text, str) and re.search(r'contradict|inconsist|contradiction', text, flags=re.IGNORECASE):
                    parsed.setdefault('detect_contradictions', {})
                    parsed['detect_contradictions']['contradictions'] = parsed['detect_contradictions'].get('contradictions', []) + ['detected_in_text']
                    parsed['detect_contradictions']['flag'] = 1

            # detect risky phrases like opened new lines of credit, recent bankrupt, default, late
            risky = parsed.get('extract_risky') or {}
            rp_list = risky.get('risky_phrases', []) if isinstance(risky, dict) else []
            # common risky indicators
            patterns = [r'open(ed)?\s+.*new\s+lines?\s+of\s+credit', r'new\s+lines?\s+of\s+credit', r'bankrupt', r'default', r'late payment', r'collection']
            for p in patterns:
                if isinstance(text, str) and re.search(p, text, flags=re.IGNORECASE):
                    cand = re.search(p, text, flags=re.IGNORECASE).group(0)
                    if cand not in rp_list:
                        rp_list.append(cand)

            if rp_list:
                parsed.setdefault('extract_risky', {})
                parsed['extract_risky']['risky_phrases'] = rp_list
                parsed['extract_risky']['count'] = len(rp_list)

        except Exception:
            return parsed
        return parsed

    # apply augmentation using the original free-text when available
    try:
        text_notes = df_row.get('text_notes') or df_row.get('text') or ''
        parsed = _augment_parsed_with_text(parsed, text_notes)
    except Exception:
        pass

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

    # Debug: when running with real LLM (mock=False), print parsed/raw to help diagnose misclassifications
    try:
        if not mock:
            print(f"DEBUG: applicant={applicant_id} parsed={parsed} features={features}")
    except Exception:
        pass

    print(f"DEBUG run_feature_extraction END applicant={applicant_id} -> risky_count={risky_count} sentiment={sentiment_score} mock={mock}")
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

import inspect
import json
import traceback
from typing import Any, Dict

# Try to import the project's llm_handler implementation(s)
try:
    # Prefer package import if available
    from llms.backend import llm_handler as _lh
except Exception:
    # Fallback to dynamic import by file (integrations already has similar logic);
    # we'll try a simple import path and otherwise handle gracefully.
    try:
        import importlib
        _lh = importlib.import_module('llms.backend.llm_handler')
    except Exception:
        _lh = None


def _log(msg: str) -> None:
    try:
        print(f"LLM_COMPAT: {msg}")
    except Exception:
        pass


def call_llm(prompt: str, mode: str = "summary", temperature: float = 0.0, use_cache: bool = True, mock: bool = False, **kwargs) -> Dict[str, Any]:
    """
    Canonical wrapper around the project's available llm handler implementations.

    This function attempts to call the underlying `llm_handler.call_llm` in a
    compatible way across variants that accept different parameters. It is
    intentionally defensive and non-breaking: if the concrete handler does not
    accept `mode` as a kwarg, the desired mode is embedded into the prompt.
    """
    if _lh is None:
        _log("No llm_handler module found; returning mock response if available.")
        # Try to return a mock if available
        try:
            return call_llm_mock(prompt)
        except Exception:
            raise RuntimeError("No LLM handler available in this environment.")

    func = getattr(_lh, "call_llm", None)
    if func is None:
        _log("Underlying handler has no 'call_llm' function; attempting 'call_llm_mock'.")
        func = getattr(_lh, "call_llm_mock", None)
        if func is None:
            raise RuntimeError("No usable call_llm or call_llm_mock in llm_handler.")

    # Inspect signature
    try:
        sig = inspect.signature(func)
        params = sig.parameters
    except Exception:
        sig = None
        params = {}

    _log(f"Detected llm_handler module at={getattr(_lh, '__file__', str(_lh))} signature={sig}")

    # Prepare prompt: embed mode when handler doesn't accept a 'mode' kwarg
    prompt_with_mode = f"Mode: {mode}\n" + prompt

    # Build kwargs for call where safe
    call_kwargs = {}
    if 'temperature' in params:
        call_kwargs['temperature'] = float(temperature)
    if 'use_cache' in params:
        call_kwargs['use_cache'] = bool(use_cache)
    if 'mock' in params:
        call_kwargs['mock'] = bool(mock)

    # If handler accepts 'mode', pass it; otherwise embed in prompt
    if 'mode' in params:
        call_kwargs['mode'] = mode

    # Merge any additional kwargs passed by callers into call_kwargs
    for k, v in kwargs.items():
        if k in params:
            call_kwargs[k] = v

    # Attempt calls with decreasing assumptions about signature
    attempts = []
    try:
        if call_kwargs:
            _log(f"Calling underlying handler with kwargs: {list(call_kwargs.keys())}")
            return func(prompt_with_mode if 'mode' not in call_kwargs else prompt, **call_kwargs)
        else:
            _log("Calling underlying handler with prompt-only (mode embedded in prompt).")
            return func(prompt_with_mode)
    except TypeError as te:
        attempts.append(str(te))
        # Fallback: try prompt-only
        try:
            _log("Fallback: calling underlying handler with prompt-only.")
            return func(prompt_with_mode)
        except Exception as e:
            attempts.append(str(e))
            # Last resort: try raw prompt_base
            try:
                _log("Final fallback: calling underlying handler with raw prompt.")
                return func(prompt)
            except Exception as e2:
                attempts.append(str(e2))
                _log("All call attempts to underlying handler failed: " + json.dumps(attempts))
                # If there's a mock available, use it
                mock_func = getattr(_lh, 'call_llm_mock', None)
                if callable(mock_func):
                    _log("Using handler's mock implementation as fallback.")
                    return mock_func(prompt)
                # Re-raise the original TypeError for visibility
                raise


# Expose mock wrapper for convenience if underlying module provides one
def call_llm_mock(prompt: str) -> Dict[str, Any]:
    if _lh is not None and hasattr(_lh, 'call_llm_mock'):
        return getattr(_lh, 'call_llm_mock')(prompt)
    # Basic deterministic mock if none provided
    return {
        "summary": f"MOCK: {prompt[:120]}",
        "sentiment_score": 0.0,
        "risky_phrase_count": 0,
        "contradiction_flag": 0,
        "credibility_score": 0.5,
    }

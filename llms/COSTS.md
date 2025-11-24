# COSTS.md
This file records expected API cost estimates and demo recommendations.

- Calls per applicant: 4 LLM prompts (summary, extract_risky, detect_contradictions, sentiment).
- If using an LLM priced like GPT-4-class, estimate 0.03 - 0.12 USD per 1K tokens (verify provider pricing).
- Typical demo prompt size: ~100-400 tokens per prompt depending on applicant text length and prompt template.
- Conservative demo estimate: 4 prompts * 300 tokens = 1200 tokens per applicant => roughly 0.036 - 0.144 USD per applicant (approx).
- Use mock mode and caching (`artifacts/llm_cache.json`) to avoid repeated calls and reduce cost during development.
- Track cost per run in a real `COSTS_LOG.md` if you want to record per-applicant actual token usage (recommended for audits).

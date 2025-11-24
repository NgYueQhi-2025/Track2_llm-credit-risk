"""LLM handler: supports real OpenAI and mock mode + simple JSON response parsing + file-based caching.
if mode == 'summary':
return json.dumps({"summary": "Applicant describes entrepreneurial experience; moderate financial detail.", "confidence": 0.82})
if mode == 'extract_risky':
return json.dumps({"risky_phrases": ["late payments", "multiple loans"], "count": 2})
if mode == 'detect_contradictions':
return json.dumps({"contradictions": ["claimed revenue inconsistent with bank statement"], "flag": 1})
if mode == 'sentiment':
return json.dumps({"sentiment": "neutral", "score": 0.05})
return json.dumps({"raw": "mocked"})


def call_llm(self, prompt: str, mode: str = 'summary', temperature: float = 0.0, use_cache: bool = True) -> str:
"""Call the LLM (real or mock). Returns raw string output.
Caching: keyed by SHA256(prompt + mode)
"""
key = _hash_input(prompt + '::' + mode)
cache = _load_cache()
if use_cache and key in cache:
return cache[key]


if self.mode == 'mock':
out = self._mock_response(prompt, mode)
else:
out = self._call_openai(prompt, temperature=temperature)


# Save to cache
cache[key] = out
_save_cache(cache)
return out


def parse_response(self, raw: str, mode: str = 'summary') -> Dict[str, Any]:
"""Parse LLM text output. We ask LLM to return JSON in prompts, but be defensive.
Returns a dictionary with expected keys for each mode.
"""
try:
parsed = json.loads(raw)
return parsed
except Exception:
# Best-effort fallback: try to extract lines like key: value
out = {}
for line in raw.splitlines():
if ':' in line:
k, v = line.split(':', 1)
out[k.strip()] = v.strip()
if out:
return out
return {"raw": raw}




# Utility: load prompt text from backend/prompts
def load_prompt_template(name: str) -> str:
ppath = os.path.join(os.path.dirname(__file__), 'prompts', name + '.txt')
with open(ppath, 'r', encoding='utf-8') as f:
return f.read()




if __name__ == '__main__':
# Quick demo
lh = LLMHandler(mode='mock')
template = load_prompt_template('summary')
applicant_text = "I run a small online shop, revenue roughly $2k/month. I had one late payment last year."
prompt = template.replace('{text}', applicant_text)
raw = lh.call_llm(prompt, mode='summary')
print('RAW:', raw)
print('PARSED:', lh.parse_response(raw, mode='summary'))

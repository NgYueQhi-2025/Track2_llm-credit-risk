import requests, os, json
base = os.getenv('LOCAL_LLM_URL','http://127.0.0.1:11434').rstrip('/')
endpoints = ['','/health','/generate','/api/generate','/chat','/api/chat','/v1/chat/completions']
for ep in endpoints:
    url = base + ep
    try:
        r = requests.get(url, timeout=3)
        print(url, '->', r.status_code, r.text[:200])
    except Exception as e:
        print(url, '->', 'ERROR', repr(e))

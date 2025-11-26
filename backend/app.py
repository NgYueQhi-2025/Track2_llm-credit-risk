# --- snippet to integrate into your upload/processing flow ---
# app.py
import os
import sys

# 1. Get the directory of the current file (backend)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Get the project root directory (go up one level from backend)
project_root = os.path.join(current_dir, '..')

# 3. Add the project root to the list of places Python looks for modules
if project_root not in sys.path:
    sys.path.insert(0, project_root)


USE_MOCK = (os.environ.get('USE_MOCK_LLM','true').lower() == 'true')
CHUNK_SIZE = 10  # recommended default: 5-50 words per LLM call for balance of cost & resolution


# now risky_phrases & total_count can be used to create numeric features
sentiment_score = ...  # if you call mode='sentiment' similarly and aggregate

# display in Streamlit
# st.write("Detected risky phrases:", risky_phrases)
# st.write("Risk count:", total_count)

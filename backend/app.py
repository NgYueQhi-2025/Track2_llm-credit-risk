# --- snippet to integrate into your upload/processing flow ---

from llms.backend.llm_handler import process_text_word_by_word
import os

USE_MOCK = (os.environ.get('USE_MOCK_LLM','true').lower() == 'true')
CHUNK_SIZE = 10  # recommended default: 5-50 words per LLM call for balance of cost & resolution

# suppose `full_text` is the text you extracted from uploaded pdf/jpg/png/csv
# e.g., using pdfplumber or pytesseract before this call:
word_by_word_results = process_text_word_by_word(
    text=full_text,
    mode='extract_risky',
    chunk_size_words=CHUNK_SIZE,
    temperature=0.0,
    mock=USE_MOCK
)

# Example aggregation: collect all risky phrases and total counts
risky_phrases = []
total_count = 0
for item in word_by_word_results:
    if item.get("parsed") and isinstance(item["parsed"], dict):
        rp = item["parsed"].get("risky_phrases") or []
        cnt = item["parsed"].get("count") or 0
        risky_phrases.extend(rp)
        total_count += cnt

# now risky_phrases & total_count can be used to create numeric features
sentiment_score = ...  # if you call mode='sentiment' similarly and aggregate

# display in Streamlit
# st.write("Detected risky phrases:", risky_phrases)
# st.write("Risk count:", total_count)

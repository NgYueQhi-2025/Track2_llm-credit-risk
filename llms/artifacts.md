✅ What is the artifacts/ folder?

In ML + LLM projects, artifacts means:
files that are generated automatically during runtime or training — not handwritten code.

They are outputs of your system.

Think of the artifacts/ folder as the “storage room” for everything your backend produces and needs to reuse later.

📁 Breakdown of Each File in /artifacts
1. llm_cache.json

Created at runtime by llm_handler.py

Every time your system calls the LLM API (OpenAI or mock), it stores the input hash + LLM output.

On the next identical request, instead of paying the API again, it loads the result from cache.

This reduces cost, improves speed, and makes your system reproducible.

You do NOT create this file manually — your code creates it automatically.

2. features.csv

Created at runtime by feature_extraction.py

This file stores the final numeric features extracted from:

LLM outputs

applicant structured data

Example columns:

applicant_id	sentiment_score	risky_phrase_count	contradiction_flag	credibility_score

This dataset is what you feed into machine learning during training.

3. model.pkl

Generated after running backend/train_model.py

This file stores the trained ML model using Python's joblib.dump().

In your case:

Logistic Regression model

RandomForest model

(or whichever model you choose to save)

Your Flask API loads this file to make predictions inside /score.

4. scaler.pkl

Generated after training along with the model

This file stores the StandardScaler or MinMaxScaler, which ensures:

features used during inference

are scaled the same way as during training

If you do not use the same scaler, your predictions will be wrong.

🧠 Summary Table
File	Who creates it?	When?	Purpose
llm_cache.json	LLM handler	When calling LLM	Saves LLM outputs to avoid repeated API calls
features.csv	Feature extractor	When extracting features	Stores numeric features for model training
model.pkl	Training script	After training	Stores trained ML model
scaler.pkl	Training script	After training	Stores feature scaler
📌 Should you commit these to GitHub?
File	Commit to GitHub?	Reason
llm_cache.json	❌ No	Changes often, API content, not stable
features.csv	⚠️ Optional	Only commit synthetic data; not real financial data
model.pkl	❌ No	Should be regenerated per update
scaler.pkl	❌ No	Same reason as model.pkl

Only the folders, not the generated files, should be tracked.

from flask import Flask, request, jsonify
from backend.llm_handler import call_llm
from backend.feature_extraction import extract_features
from backend.train_model import train_model
import joblib
import os

app = Flask(__name__)

MODEL_PATH = "artifacts/model.pkl"
SCALER_PATH = "artifacts/scaler.pkl"


# ------------------------------------------------------------
# Health Check
# ------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "API is running"}), 200


# ------------------------------------------------------------
# Test LLM Integration (Useful for debugging)
# ------------------------------------------------------------
@app.route("/llm-test", methods=["POST"])
def llm_test():
    data = request.json
    text = data.get("text", "")
from flask import Flask, request, jsonify
from backend.llm_handler import call_llm
from backend.feature_extraction import extract_features
from backend.train_model import train_model
import joblib
import os

app = Flask(__name__)

MODEL_PATH = "artifacts/model.pkl"
SCALER_PATH = "artifacts/scaler.pkl"


# ------------------------------------------------------------
# Health Check
# ------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "API is running"}), 200


# ------------------------------------------------------------
# Test LLM Integration (Useful for debugging)
# ------------------------------------------------------------
@app.route("/llm-test", methods=["POST"])
def llm_test():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "Missing 'text' in request"}), 400

    try:
        response = call_llm(prompt=text, mode="summary", temperature=0)
        return jsonify({"llm_output": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------
# Extract numeric features (LLM output → features)
# ------------------------------------------------------------
@app.route("/extract-features", methods=["POST"])
def extract_features_api():
    data = request.json

    text = data.get("text")
    metadata = data.get("metadata", {})

    if not text:
        return jsonify({"error": "Missing 'text'"}), 400

    try:
        llm_output = call_llm(prompt=text, mode="summary", temperature=0)
        features = extract_features(llm_output, metadata)

        return jsonify({"features": features}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------
# Score Applicant (Main Endpoint)
# ------------------------------------------------------------
@app.route("/score", methods=["POST"])
def score():
    data = request.json

    text = data.get("text")
    metadata = data.get("metadata", {})

    if not text:
        return jsonify({"error": "Missing 'text'"}), 400

    # Step 1: load model & scaler
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return jsonify({"error": "Model not trained. Call /train first."}), 500

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    try:
        # Step 2: LLM → features
        llm_output = call_llm(prompt=text, mode="summary", temperature=0)
        features_dict = extract_features(llm_output, metadata)

        # Convert dict → list in consistent feature order
        feature_vector = [
            features_dict["sentiment_score"],
            features_dict["risky_phrase_count"],
            features_dict["contradiction_flag"],
            features_dict["credibility_score"],
        ]

        # Step 3: scale features
        vector_scaled = scaler.transform([feature_vector])

        # Step 4: predict risk
        risk_score = float(model.predict_proba(vector_scaled)[0][1])

        return jsonify({
            "risk_score": risk_score,
            "model": type(model).__name__,
            "features_used": features_dict
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------
# Train Model
# ------------------------------------------------------------
@app.route("/train", methods=["POST"])
def train():
    try:
        train_model()
        return jsonify({"message": "Model trained successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------
# Run Flask App
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

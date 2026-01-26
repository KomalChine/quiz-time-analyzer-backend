from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# ============================
# LOAD ML MODELS
# ============================

quiz_personal_model = joblib.load(
    "models/quiz_personal_model.pkl"
)

difficulty_model = joblib.load(
    "models/difficulty_model.pkl"
)

print("✅ ML models loaded successfully")

# ============================
# HELPER: ASSIGN TIME LIMIT
# ============================

def get_time_limit(difficulty):
    if difficulty == "easy":
        return 10      # seconds
    elif difficulty == "medium":
        return 20
    elif difficulty == "hard":
        return 30
    else:
        return None

# ============================
# ML INFERENCE API
# ============================

@app.route("/api/predict-question", methods=["POST"])
def predict_question():
    """
    Expected JSON payload:
    {
      "question_text": "...",
      "question_length": 45,
      "word_count": 8,
      "option_count": 4,
      "has_options": 1,
      "avg_option_length": 6,
      "option_text_present": 1,
      "position_index": 5
    }
    """

    data = request.json

    # Convert to DataFrame (ML models expect DataFrame)
    df = pd.DataFrame([data])

    # --------------------
    # MODEL 1: Quiz vs Personal
    # --------------------
    q_type = quiz_personal_model.predict(df)[0]

    response = {
        "question_type": q_type
    }

    # --------------------
    # MODEL 2: Difficulty (ONLY if quiz)
    # --------------------
    if q_type == "quiz":
        difficulty = difficulty_model.predict(df)[0]
        time_limit = get_time_limit(difficulty)

        response.update({
            "difficulty": difficulty,
            "time_limit": time_limit
        })

    return jsonify(response)

# ============================
# RUN SERVER
# ============================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

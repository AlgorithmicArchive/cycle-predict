import os
from flask import Flask, request, jsonify
import pickle
import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "cycle_prediction_model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        cycle_length_lag1 = data["cycle_length_lag1"]
        cycle_length_lag2 = data["cycle_length_lag2"]
        last_end_date = data["last_end_date"]

        input_df = pd.DataFrame(
            [{"cycle_length_lag1": cycle_length_lag1, "cycle_length_lag2": cycle_length_lag2}]
        )
        predicted_after_days = model.predict(input_df)[0]

        last_end_date = datetime.strptime(last_end_date, "%Y-%m-%d")
        next_start_date = last_end_date + timedelta(days=predicted_after_days)

        return jsonify(
            {
                "predicted_startDay": next_start_date.day,
                "predicted_startMonth": next_start_date.month,
                "predicted_startYear": next_start_date.year,
                "predicted_afterDays": predicted_after_days,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

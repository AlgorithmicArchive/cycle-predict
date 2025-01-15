from flask import Flask, request, jsonify
import pickle
import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)

# Load the trained model
with open('cycle_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Expecting JSON input
    try:
        # Extract input data
        cycle_length_lag1 = data['cycle_length_lag1']
        cycle_length_lag2 = data['cycle_length_lag2']
        last_end_date = data['last_end_date']  # "YYYY-MM-DD" format

        # Prepare input for the model
        input_df = pd.DataFrame([{
            'cycle_length_lag1': cycle_length_lag1,
            'cycle_length_lag2': cycle_length_lag2
        }])

        # Predict next cycle length
        predicted_after_days = model.predict(input_df)[0]

        # Calculate next start date
        last_end_date = datetime.strptime(last_end_date, '%Y-%m-%d')
        next_start_date = last_end_date + timedelta(days=predicted_after_days)

        # Return the prediction
        return jsonify({
            'predicted_startDay': next_start_date.day,
            'predicted_startMonth': next_start_date.month,
            'predicted_startYear': next_start_date.year,
            'predicted_afterDays': predicted_after_days
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

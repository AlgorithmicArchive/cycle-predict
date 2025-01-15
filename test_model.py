import pandas as pd
import pickle
from datetime import datetime, timedelta

def predict_next_cycle(last_cycle):
    # Load the trained model
    with open('cycle_prediction_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Prepare the input data for prediction (lagging features)
    input_data = {
        'cycle_length_lag1': last_cycle['afterDays'],  # Most recent cycle length
        'cycle_length_lag2': last_cycle.get('previousAfterDays', last_cycle['afterDays'])  # Second most recent cycle
    }
    input_df = pd.DataFrame([input_data])

    # Predict the next cycle's `afterDays`
    predicted_after_days = model.predict(input_df)[0]

    # Calculate the last cycle's end date
    last_end_date = datetime(
        year=last_cycle['endYear'],
        month=last_cycle['endMonth'],
        day=last_cycle['endDay']
    )

    # Predict the next start date
    next_start_date = last_end_date + timedelta(days=predicted_after_days)

    # Return the predicted start date and predicted cycle length
    return {
        'predicted_startDay': next_start_date.day,
        'predicted_startMonth': next_start_date.month,
        'predicted_startYear': next_start_date.year,
        'predicted_afterDays': predicted_after_days
    }

# Example Usage
if __name__ == '__main__':
    # Last known cycle
    last_cycle = {
        "startDay": 25,
        "startMonth": 7,
        "startYear": 2024,
        "endDay": 31,
        "endMonth": 7,
        "endYear": 2024,
        "afterDays": 26,
        "previousAfterDays": 21  # Add a previous cycle length if available
    }

    # Predict the next cycle start date
    prediction = predict_next_cycle(last_cycle)
    print("Predicted Next Cycle Start Date:")
    print(f"Day: {prediction['predicted_startDay']}, "
          f"Month: {prediction['predicted_startMonth']}, "
          f"Year: {prediction['predicted_startYear']}")
    print(f"Predicted Cycle Length: {prediction['predicted_afterDays']:.2f} days")

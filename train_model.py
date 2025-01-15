import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle

def train_model():
    # Load the dataset (replace 'data.json' with your actual data path)
    df = pd.read_json('data.json')

    # Ensure the required fields are extracted correctly
    df['start_date'] = pd.to_datetime(
        dict(year=df['startYear'], month=df['startMonth'], day=df['startDay'])
    )
    df = df.sort_values(by='start_date')  # Sort by start date

    # Check for missing or inconsistent data in 'afterDays'
    if df['afterDays'].isnull().any():
        raise ValueError("Missing values detected in 'afterDays' column.")

    # Create lagging features
    df['cycle_length_lag1'] = df['afterDays'].shift(1)
    df['cycle_length_lag2'] = df['afterDays'].shift(2)

    # Drop rows with NaN values due to lagging
    df = df.dropna()

    # Prepare training data
    X = df[['cycle_length_lag1', 'cycle_length_lag2']]
    y = df['afterDays']

    # Train a Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save the trained model
    with open('cycle_prediction_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Model training completed and saved as 'cycle_prediction_model.pkl'.")

if __name__ == '__main__':
    train_model()

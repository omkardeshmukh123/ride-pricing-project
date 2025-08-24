from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd

# Create the app object
app = FastAPI(title="Dynamic Pricing API")

# Load the trained model and scaler
# Ensure these files are in the same directory as your main.py
try:
    model = joblib.load('rf_model.joblib')
    scaler = joblib.load('scaler.joblib')
    # This file should contain the column order the model was trained on
    training_cols = joblib.load('training_columns.joblib') 
except FileNotFoundError:
    model = None
    scaler = None
    training_cols = []
    print("Error: Model or scaler files not found. The API will not work.")


# Define the input data model using Pydantic
class RideInput(BaseModel):
    distance_km: float
    temperature_celsius: float
    cab_type: str
    weather_condition: str
    timestamp: str 

# Define a root endpoint to prevent 404 errors on browser access
@app.get("/")
def read_root():
    return {"message": "Welcome to the Dynamic Pricing API. Use the /docs endpoint to test."}


@app.post("/predict")
def predict_price(ride_input: RideInput):
    # Ensure model and scaler are loaded
    if not all([model, scaler, training_cols]):
        return {"error": "Model not loaded. Please check server logs."}

    # 1. Convert input Pydantic model to a DataFrame
    input_df = pd.DataFrame([ride_input.dict()])

    # 2. Perform Feature Engineering on the timestamp
    input_df['timestamp'] = pd.to_datetime(input_df['timestamp'])
    input_df['hour_of_day'] = input_df['timestamp'].dt.hour
    input_df['day_of_week'] = input_df['timestamp'].dt.dayofweek
    input_df['is_weekend'] = (input_df['timestamp'].dt.dayofweek >= 5).astype(int)
    input_df = input_df.drop('timestamp', axis=1)

    # 3. Separate numerical and categorical features
    numerical_features = input_df.select_dtypes(include=['float64', 'int64', 'int32']).columns.tolist()
    categorical_features = input_df.select_dtypes(include=['object']).columns.tolist()

    # 4. Scale the numerical features ONLY
    # Note: The scaler expects a DataFrame, so we use [[]]
    scaled_numerical_df = pd.DataFrame(scaler.transform(input_df[numerical_features]), columns=numerical_features)

    # 5. Perform One-Hot Encoding on categorical features
    one_hot_encoded_df = pd.get_dummies(input_df[categorical_features], columns=categorical_features)

    # 6. Combine scaled numerical and one-hot encoded features
    processed_df = pd.concat([scaled_numerical_df, one_hot_encoded_df], axis=1)
    
    # 7. Align columns with the model's training columns
    # Reindex the processed_df to have the same columns, filling missing ones with 0
    final_df = processed_df.reindex(columns=training_cols, fill_value=0)

    # 8. Make a prediction
    prediction = model.predict(final_df)

    # 9. Return the prediction
    return {"predicted_demand_multiplier": float(prediction[0])}
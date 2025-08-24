from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd

# 1. Create the app object
app = FastAPI(title="Dynamic Pricing API")

# 2. Load the trained model, scaler, and training columns
model = joblib.load('rf_model.joblib')
scaler = joblib.load('scaler.joblib')
training_cols = joblib.load('training_columns.joblib') # <-- The fix is here

# 3. Define the input data model using Pydantic
class RideInput(BaseModel):
    distance_km: float
    temperature_celsius: float
    cab_type: str
    weather_condition: str
    timestamp: str # Example: "2025-08-23 19:30:00"

# Add a root endpoint for basic checks
@app.get("/")
def read_root():
    return {"message": "Welcome to the Dynamic Pricing API. Use the /docs endpoint to test."}

@app.post("/predict")
def predict_price(ride_input: RideInput):
    # 1. Convert input Pydantic model to a DataFrame
    input_df = pd.DataFrame([ride_input.dict()])

    # 2. Perform Feature Engineering
    input_df['timestamp'] = pd.to_datetime(input_df['timestamp'])
    input_df['hour_of_day'] = input_df['timestamp'].dt.hour
    input_df['day_of_week'] = input_df['timestamp'].dt.dayofweek
    input_df['is_weekend'] = (input_df['timestamp'].dt.dayofweek >= 5).astype(int)
    input_df = input_df.drop('timestamp', axis=1)

    # 3. Perform One-Hot Encoding
    input_df = pd.get_dummies(input_df, columns=['cab_type', 'weather_condition'])
    
    # 4. Align columns with the model's training columns
    # This is a crucial step to ensure the input has the exact same columns as the training data
    input_df = input_df.reindex(columns=training_cols, fill_value=0)

    # 5. Scale the features
    # The scaler was trained on the same columns, so this will work correctly now
    scaled_features = scaler.transform(input_df)

    # 6. Make a prediction
    prediction = model.predict(scaled_features)

    # 7. Return the prediction
    return {"predicted_demand_multiplier": float(prediction[0])}
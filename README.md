# Dynamic Ride-Sharing Pricing API 🚕

## 1. Project Description

A machine learning project that predicts dynamic price multipliers for a ride-sharing service based on real-time factors. This project covers the complete end-to-end data science workflow, from data exploration and feature engineering to model training and deployment as a live API.

---

## 2. Features

* **End-to-End ML Pipeline:** Covers data cleaning, preprocessing, and feature engineering.
* **Advanced Modeling:** Uses a Random Forest Regressor to predict price multipliers.
* **Live API:** The trained model is deployed using a FastAPI server.
* **Interactive Docs:** Automatically generated, interactive API documentation via Swagger UI.

---

## 3. Tech Stack

* **Language:** Python
* **Data Science Libraries:** Pandas, NumPy, Scikit-learn
* **API Framework:** FastAPI, Uvicorn
* **Model Serialization:** Joblib
* **Development:** JupyterLab, Visual Studio Code

---

## 4. Setup and Installation

To run this project locally, follow these steps:

1.  Clone the repository:
    ```bash
    git clone <your-github-repo-link>
    cd <your-project-folder>
    ```
2.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## 5. Usage

1.  Run the API server from the root directory:
    ```bash
    uvicorn main:app --reload
    ```
2.  Open your web browser and go to `http://127.0.0.1:8000`. You should see the welcome message.

3.  To get a prediction, go to the interactive documentation at `http://12_7.0.0.1:8000/docs`.
    * Expand the `/predict` endpoint.
    * Click "Try it out."
    * Fill in the request body with ride details and click "Execute."

    **Example Request Body:**
    ```json
    {
      "distance_km": 10.5,
      "temperature_celsius": 32.0,
      "cab_type": "Sedan",
      "weather_condition": "Rainy",
      "timestamp": "2025-08-24 19:30:00"
    }
    ```
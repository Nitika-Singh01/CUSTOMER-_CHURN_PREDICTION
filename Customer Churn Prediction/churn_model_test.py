# test_model.py
# This file is used to test the saved churn prediction model without needing a Streamlit app.
# You can run this to verify predictions manually or for demo/testing purposes.

import pandas as pd
import joblib

# Load the trained pipeline (with preprocessing + model)
model = joblib.load("xgb_churn_pipeline.pkl")

# Sample data in the same format as training data (dictionary for manual entry or use pd.read_csv)
sample_input = pd.DataFrame([
    {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 5,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'DSL',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 70.35,
        'TotalCharges': 350.5
    }
])

# Make prediction
prediction = model.predict(sample_input)

# Output result
print("Prediction (1 = Churn, 0 = Not Churn):", prediction[0])

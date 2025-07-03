# CUSTOMER-_CHURN_PREDICTION
Customer Churn Prediction  A machine learning pipeline using XGBoost and scikit-learn to predict customer churn based on user attributes. Supports CSV input and direct model testing via Python script.

This project builds a machine learning pipeline to predict customer churn using the XGBoost algorithm. It includes data preprocessing, model training, and a script to test the model with a sample input.

Features
Uses XGBoost for high-performance classification

Preprocessing with scikit-learn pipelines

Supports both single input and batch CSV prediction

Easily testable using test_model.py

Files Included
xgb_churn_pipeline.pkl – Trained model pipeline

test_model.py – Script to test the model with a CSV file

sample_input.csv – Sample input file for testing

requirements.txt – Python dependencies

train_churn_model.ipynb – (Optional) Colab notebook used to build and export the model

Model
The model is trained using XGBoost on a labeled dataset. Preprocessing includes encoding categorical variables and scaling numerical ones using a ColumnTransformer inside a Pipeline.

Requirements
Python 3.8+,scikit-learn 1.4.2, xgboost, pandas,numpy, joblib












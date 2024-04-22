# Predict Customer Churn

This repository contains the code for the project **Predict Customer Churn**, which is part of the ML DevOps Engineer Nanodegree program by Udacity.

## Project Description
This project aims to develop a machine learning model to predict customer churn for a credit card company. By analyzing historical data, the model identifies patterns and characteristics that indicate the likelihood of customers discontinuing their service. The project incorporates best practices in software engineering and machine learning operations to ensure the model's robustness and scalability.

## Files and Data Description
- `churn_library.py`: Main Python script containing functions for data loading, preprocessing, feature engineering, model training, and evaluation.
- `data/`:
  - `bank_data.csv`: Dataset used for training the churn prediction model. Contains customer demographics, transaction data, and churn labels.
- `models/`:
  - `rfc_model.pkl`: Saved Random Forest model after training.
  - `logistic_model.pkl`: Saved Logistic Regression model after training.
- `plots/`: Folder containing visualizations generated during exploratory data analysis.
- `README.md`: This file, providing an overview and instructions for the project.

## Running Files
To run the files and execute the churn prediction pipeline, follow these steps:
1. Ensure Python 3.8+ is installed on your machine.
2. Install required packages:
   ```bash
   pip install -r requirements.txt

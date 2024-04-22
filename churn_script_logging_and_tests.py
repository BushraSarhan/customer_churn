"""
Churn Script for Logging and Testing

This script tests the functionality of the `churn_library.py` functions by conducting
unit tests for each function to ensure they operate correctly and logging the results,
including any errors encountered during execution. It is crucial for ensuring the
robustness and reliability of the churn prediction library.

Usage:
Run this script from the command line to execute all tests and output logs to the
`/logs` folder.

Author: Bushra Bin Sarhan
Date: April 9, 2024.
"""

import logging
import os
from sklearn.model_selection import train_test_split
from churn_library import import_data, perform_eda, encoder_helper, perform_feature_engineering, train_models

# Setup logging configuration
if not os.path.exists('./logs'):
    os.makedirs('./logs')
logging.basicConfig(
    filename='./logs/churn_testing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')

DATA_FILE_PATH = "./data/bank_data.csv"


def test_load_data():
    """
    Test function to load data from a CSV file and ensure the DataFrame is not empty.

    Raises:
        AssertionError: If the loaded DataFrame is empty.
    """
    try:
        data_frame = import_data(DATA_FILE_PATH)
        assert not data_frame.empty, "Data frame is empty."
        logging.info("Data loading test passed.")
    except AssertionError as e:
        logging.error("Assertion error in test_load_data: %s", str(e))
    except Exception as e:
        logging.error("Exception in test_load_data: %s", str(e))


def test_explore_data():
    """
    Test function to perform exploratory data analysis and save visualization plots.

    Raises:
        AssertionError: If errors occur during exploratory data analysis.
    """
    try:
        data_frame = import_data(DATA_FILE_PATH)
        perform_eda(data_frame)
        logging.info("Explore data test passed.")
    except Exception as e:
        logging.error("Exception in test_explore_data: %s", str(e))


def test_feature_engineering():
    """
    Test function to perform feature engineering and check if it correctly creates train and test datasets.
    """
    try:
        data_frame = import_data(DATA_FILE_PATH)
        data_frame = encoder_helper(data_frame, ['Gender'], 'Churn')
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            data_frame, 'Churn')
        assert not X_train.empty and not X_test.empty, "Feature engineering output is empty."
        logging.info("Feature engineering test passed.")
    except Exception as e:
        logging.error("Exception in test_feature_engineering: %s", str(e))


def test_train_models():
    """
    Test function to train models and verify absence of errors.
    """
    try:
        data_frame = import_data(DATA_FILE_PATH)
        data_frame = encoder_helper(data_frame, ['Gender'], 'Churn')
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            data_frame, 'Churn')
        train_models(X_train, X_test, y_train, y_test)
        logging.info("Model training test passed.")
    except Exception as e:
        logging.error("Exception in test_train_models: %s", str(e))


def run_all_tests():
    """
    Function to run all test functions and log the results.
    """
    test_load_data()
    test_explore_data()
    test_feature_engineering()
    test_train_models()
    logging.info("All tests completed.")


if __name__ == "__main__":
    run_all_tests()

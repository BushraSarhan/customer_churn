"""
Churn Library for Credit Card Customers

This library processes and analyzes credit card customer data to predict churn.
It includes functions for data preparation, exploratory data analysis, feature
engineering, model training, and evaluation. Designed to follow PEP 8 standards
and software engineering best practices, it offers a structured approach for
churn prediction projects.

Usage:
Refer to README.md for setup and execution instructions.

Author: Bushra Bin Sarhan
Date: April 9, 2024.
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot

# Additional settings for headless servers
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sns.set(style='darkgrid')  # Configure seaborn's style here


def import_data(file_path):
    '''
    returns dataframe for the csv found at file_path

    input:
            file_path: a path to the csv
    output:
            data_frame: pandas dataframe
    '''
    data_frame = pd.read_csv(file_path)
    return data_frame


def perform_eda(data_frame):
    '''
    perform eda on data_frame and save figures to images folder
    input:
            data_frame: pandas dataframe

    output:
            None
    '''
    # Ensure the images/eda directory exists
    if not os.path.exists('images/eda'):
        os.makedirs('images/eda')

    # Check if 'Attrition_Flag' exists in the DataFrame
    if 'Attrition_Flag' not in data_frame.columns:
        raise KeyError(
            "Column 'Attrition_Flag' is missing. Please check your data input.")

    # Convert 'Attrition_Flag' to a numerical 'Churn' column and remove the
    # original column
    data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
        lambda x: 0 if x == 'Existing Customer' else 1)
    data_frame.drop(columns=['Attrition_Flag'], inplace=True)

    # Visualization: Histogram of Churn
    plt.figure(figsize=(20, 10))
    data_frame['Churn'].hist()
    plt.savefig('images/eda/churn_histogram.png')
    plt.close()

    # Plot and save histogram of Customer Age
    plt.figure(figsize=(20, 10))
    data_frame['Customer_Age'].hist()
    plt.savefig('images/eda/customer_age_histogram.png')

    # Plot and save bar chart of Marital Status
    plt.figure(figsize=(20, 10))
    data_frame['Marital_Status'].value_counts(normalize=True).plot(kind='bar')
    plt.savefig('images/eda/marital_status_bar.png')

    # Plot and save histogram of Total Transactions Count with KDE
    plt.figure(figsize=(20, 10))
    sns.histplot(data_frame['Total_Trans_Ct'], kde=True)
    plt.savefig('images/eda/total_trans_ct_hist.png')

    # Plot and save heatmap of correlations
    plt.figure(figsize=(20, 10))
    sns.heatmap(data_frame.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('images/eda/correlation_heatmap.png')

    plt.close('all')  # Close all figures to free memory


def encoder_helper(data_frame, category_lst, response):
    '''
    Helper function to turn each categorical column into a new column with
    proportion of churn for each category.

    input:
            data_frame: pandas DataFrame
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y]

    output:
            data_frame
    '''
    for column in category_lst:
        label_encoder = LabelEncoder()
        data_frame[column] = label_encoder.fit_transform(data_frame[column])
    return data_frame


def perform_feature_engineering(data_frame, response):
    '''
    Turns each categorical column into a new column with
    propotion of churn for each category.

    input:
            data_frame: pandas DataFrame
            response: string of response name [optional argument that could be used for naming variables or index y]

    output:
            X_train: X training data
            X_test: X testing data
            y_train: training response values
            y_test: test response values
    '''
    x_data = data_frame.drop([response], axis=1)
    y_data = data_frame[response]

    X_train, X_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def plot_classification_reports(
        y_true_train,
        y_pred_train_lr,
        y_pred_train_rf):
    '''
    Produces classification reports for training and testing results and stores reports as images.

    input:
            y_true_train: training true response values
            y_pred_train_lr: training predictions from logistic regression
            y_pred_train_rf: training predictions from random forest

    output:
             None
    '''
    # Calculate classification reports
    train_report_lr = classification_report(
        y_true_train, y_pred_train_lr, output_dict=True)
    train_report_rf = classification_report(
        y_true_train, y_pred_train_rf, output_dict=True)

    # Plot classification reports as images
    _, axis = plt.subplots(2, figsize=(15, 10))
    sns.heatmap(pd.DataFrame(train_report_lr).iloc[:-1,
                                                   :].T,
                annot=True,
                cmap='Blues',
                ax=axis[0]).set_title('Logistic Regression - Training')
    sns.heatmap(pd.DataFrame(train_report_rf).iloc[:-1,
                                                   :].T,
                annot=True,
                cmap='Blues',
                ax=axis[1]).set_title('Random Forest - Training')

    # Save the plot as an image
    plt.savefig('images/classification_reports.png')
    plt.close()


def feature_importance_plot(model, x_data, output_pth):
    '''
    Prints and plots the feature importances of the model

    input:
            model: model object
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    # Save the plot
    plt.savefig(output_pth)
    plt.close()


def train_models(x_train, x_test, y_train, y_test):
    '''
    Train, store model results: images + scores, and store models

    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data

    output:
              None
    '''
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)
    lrc.fit(x_train, y_train)

    # Making predictions for evaluation
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    # Assuming you want to use x_train for the example
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_train)

    print('Random Forest and Logistic Regression models trained and evaluated.')

    # Saving models
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Return necessary objects for further use
    return cv_rfc, lrc, y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr


if __name__ == "__main__":
    # Step 1: Import data
    data = import_data("data/bank_data.csv")

    # Step 2: Perform exploratory data analysis
    if 'Attrition_Flag' in data.columns:
        # Ensure Attrition_Flag exists before calling perform_eda
        perform_eda(data)
    else:
        print("Error: 'Attrition_Flag' column not found in data.")

    # Step 3: Encode all other categorical features before feature engineering
    if 'Gender' in data.columns:
        categorical_columns = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category']
        data_encoded = encoder_helper(data, categorical_columns, 'Churn')
    else:
        print("Error: Required categorical columns missing for encoding.")

    # Step 4: Perform feature engineering
    if 'Churn' in data_encoded.columns:
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            data_encoded, 'Churn')

        # Check if the variables are correctly created and contain data
        if x_train is not None and x_test is not None and y_train is not None and y_test is not None:
            # Step 5: Train models
            cv_rfc, lrc, y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr = train_models(
                x_train, x_test, y_train, y_test)  # Correctly pass all parameters

            # Step 6: Plot classification reports
            plot_classification_reports(
                y_train, y_train_preds_lr, y_train_preds_rf)

            # Step 7: Plot feature importance
            feature_importance_plot(
                cv_rfc.best_estimator_,
                x_train,
                "path_to_store_feature_importance_plot.png")
        else:
            print("Error: Required data not available for model training.")
    else:
        print("Error: 'Churn' column missing after encoding.")

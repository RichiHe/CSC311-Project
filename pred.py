"""
This Python file is example of how your `pred.py` script should
look. Your file should contain a function `predict_all` that takes
in the name of a CSV file, and returns a list of predictions.

Your `pred.py` script can use different methods to process the input
data, but the format of the input it takes and the output your script produces should be the same.

Here's an example of how your script may be used in our test file:

    from example_pred import predict_all
    predict_all("example_test_set.csv")
"""

# basic python imports are permitted
import sys
import csv
import random

# numpy and pandas are also permitted
import numpy as np
import pandas as pd
import data_preprocess
from ManualRandomForest import ManualRandomForest


def predict(row):
    """
    Helper function to make prediction for a given input row.
    This code is here for demonstration purposes only.
    """
    # randomly choose between the three GenAI models
    # NOTE: make sure to be *very* careful of the spelling/capitalization of the models!!
    prediction = random.choice(['ChatGPT', 'Claude', 'Gemini'])

    # return the prediction
    return prediction

def predict_all(filename):
    """
    Make predictions for the data in filename
    """

    # Read the file containing the test data
    X = data_preprocess.preprocess_test(filename)
    print("X shape:  ", X.shape)

    predictions = []
    # for idx, row in df.iterrows():
    #     pred = predict(row)
    #     predictions.append(pred)

    return predictions


def predict_all_by_RF(filename):
    """
    Make predictions for the data in filename
    """

    # Read the file containing the test data
    X = data_preprocess.preprocess_test(filename, 80)
    print("X shape:  ", X.shape)

    manual_rf = ManualRandomForest('rf_model_params.json')

    predictions = manual_rf.predict(X)

    return predictions


if __name__ == "__main__":
    #暂时用traning data测试
    print(predict_all_by_RF('training_data_clean.csv'))

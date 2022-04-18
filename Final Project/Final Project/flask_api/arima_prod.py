import numpy as np
import csv
import pandas as pd
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import importlib
import re
from os import path
import gzip
from collections import defaultdict
import networkx as nx
import seaborn as sns
from matplotlib import rcParams
from datetime import date
from datetime import datetime
from pandas.plotting import lag_plot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def calc_metrics(results_df):

    acc_count = 0
    # initialize counters
    FP = 0.0 # false positives
    FN = 0.0 # false negatives
    TP = 0.0 # true positives
    TN = 0.0 # true negatives

    for index, row in results_df.iterrows():
        pred_change = row['Pred_Price_Change']
        actual_change = row['Actual_Price_Change']

        # 1 - positive case & 0 - negative case
        if actual_change == pred_change:
            #TP
            if pred_change == 'Positive':
                TP += 1
            #TN
            else:
                TN += 1
        else:
            #FP - actual = 1 & pred = 0
            if pred_change == 'Positive':
                FP += 1
            #FN - actual = 0 & pred = 1
            else:
                FN += 1

        if pred_change == actual_change:
            acc_count += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    denom = 0.5 * (FP + FN)
    FScore = TP / (TP + denom)

    return accuracy, precision, recall, FScore

def calc_ARIMA(result_df, details = True):

    result_df['Previous_Price_USD'] = result_df['Price_USD'].shift(1)

    train_df, test_df = result_df[0:int(len(result_df)*0.7)], result_df[int(len(result_df)*0.7):]

    training_data = train_df['Price_USD'].values
    test_data = test_df['Price_USD'].values

    history = [x for x in training_data]
    model_predictions = []
    N_test_observations = len(test_data)

    for time_point in range(N_test_observations):
        model = ARIMA(history, order=(4,1,0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        model_predictions.append(yhat)
        true_test_value = test_data[time_point]
        history.append(true_test_value)

    MSE_error = mean_squared_error(test_data, model_predictions)
    results_df = test_df.assign(Predicted_Price_USD = model_predictions)

    results_df['Pred_Price_Difference'] = results_df['Predicted_Price_USD']  - results_df['Previous_Price_USD']
    results_df['Actual_Price_Difference'] = results_df['Price_USD']  - results_df['Previous_Price_USD']

    results_df['Pred_Price_Change'] = np.where(results_df['Pred_Price_Difference'] >= 0, 'Positive', 'Negative')
    results_df['Actual_Price_Change'] = np.where(results_df['Actual_Price_Difference'] >= 0, 'Positive', 'Negative')

    accuracy, precision, recall, FScore = calc_metrics(results_df)
    change = results_df['Pred_Price_Change'].tail(1).item()
    return change

def fetch_nft_df(df, nft_name, nft_id):
    nft_input = '(\'' + nft_name + '\', \'' + nft_id + '\')'
    result_df = df[df['Unique_id_collection'] == nft_input]
    return result_df

def calc_change_ARIMA(df, nft_name, nft_id, details = True):
    indiv_nft = fetch_nft_df(df, nft_name, nft_id)
    return calc_ARIMA(indiv_nft, details)

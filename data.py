### Imports
import time

import requests
import numpy as np
from sklearn import preprocessing

import config

### Alpha Vantage API handling
api_key = config.api_key
symbol = 'GOOGL'
interval = '15min'  # Adjust the interval as needed
outputsize = 'full'
url_prices = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={api_key}'
time_period = '40'
series_type = 'close'
url_rsi = f'https://www.alphavantage.co/query?function=RSI&symbol={symbol}&interval={interval}&time_period={time_period}&series_type={series_type}&apikey={api_key}'
url_stoch = f'https://www.alphavantage.co/query?function=STOCH&symbol={symbol}&interval={interval}&apikey={api_key}'
url_sma = f'https://www.alphavantage.co/query?function=SMA&symbol={symbol}&interval={interval}&time_period={time_period}&series_type={series_type}&apikey={api_key}'
url_ema = f'https://www.alphavantage.co/query?function=EMA&symbol={symbol}&interval={interval}&time_period={time_period}&series_type={series_type}&apikey={api_key}'

### ### ### CUSTOM FUNCTIONS
def standardize_list(list):
    """
    Given a Python list of data of 1 type, standardize it
    using scikit-learn and return a standardized Python list
    :param list:
    list -- A Python list of data, 1 dimensional
    :return:
    A standardized Python list of Data, 1 dimensional,
    having the length of list param
    """

    list_np = np.array(list).reshape(-1,1)
    scaler = preprocessing.StandardScaler().fit(list_np)
    list_np = scaler.transform(list_np)
    return list_np.reshape(list.__len__()).tolist()

def fetch_data(url, func=None):
    """
    Fetches stock data from provided url (Alpha Vintage API)
    If we do not manage to get data within COUNTER_LIMIT seconds,
    then we execute func to fetch data
    :param url: url address of an Alpha Vintage indicator
    :param func: function used to fetch data in case we exceed API call limit. None by default
    :return: data holding values of selected indicator
    """
    COUNTER_LIMIT = 90
    SLEEP_TIME_SEC = 2

    counter = 0
    while True:
        counter += SLEEP_TIME_SEC
        try:
            response = requests.get(url)
            break
        except ValueError:
            if counter > COUNTER_LIMIT:
                if func != None:
                    return func()
                print("Did not retrieve data. API call limit exceeded")
                exit()
            time.sleep(SLEEP_TIME_SEC)
    #data = response.json()
    return response.json()
    #return data[desc]

def get_features_np(feature_list):
    """
    Given a list of features, it converts data to numpy arrays ready to
    be fed into a model
    :param feature_list: Python list of features (already preprocessed)
    First entry should be closing prices
    :return: X, y numpy arrays, ready to be fed into a model
    """
    X = []
    y = []
    window_size = 20  # Adjust the window size as needed
    num_feature_classes = feature_list.__len__()
    length = feature_list[0].__len__()
    for i in range(1,num_feature_classes-1):
        length = min(length, feature_list[i].__len__())

    for i in range(length - window_size):
        feature_vector = []

        for j in range(num_feature_classes):
            feature_vector.append(feature_list[j][i:i+window_size])

        X.append(feature_vector)
        y.append(feature_list[0][i + window_size])

    X = np.array(X)
    y = np.array(y)

    return X,y

def retrieve_features(url_list, desc_list, data_point_desc_list, should_standardize):
    """
    Given a description of data and urls, it retrieves the list of
    data points we are interested in
    :param url_list: list of Alpha Vintage API urls for our data
    :param desc_list: list of names of databases
    :param data_point_desc_list: list of names of data points
    :param should_standardize: list of booleans for whether data should
    be standardized or not
    :return: list of data points
    """
    retrieved_data = []
    retrieved_data_points = []
    current_response = None
    for i in range(url_list.__len__()):
        if i == 0 or not url_list[i].__eq__(url_list[i-1]):
            current_response = fetch_data(url_list[i])

        retrieved_data.append(current_response[desc_list[i]])

        retrieved_data_points.append(
            [float(data_point[data_point_desc_list[i]])
             for data_point in retrieved_data[i].values()]
        )
        if should_standardize[i]:
            retrieved_data_points[i] = standardize_list(retrieved_data_points[i])

    return retrieved_data_points

def retrieve_data():
    """
    Main API function in data.py
    Retrieve preselected data to fed into a model
    Feature types:
    Closing prices, volumes, RSI, Stoch, SMA, EMA
    :return: model data X,y (numpy arrays).
    Processed and ready to be fed into a model
    """
    urls = []
    desc_list = []
    data_point_desc_list = []
    should_standardize = []

    urls.append(url_prices)
    urls.append(url_prices)
    urls.append(url_rsi)
    urls.append(url_stoch)
    urls.append(url_stoch)
    urls.append(url_sma)
    urls.append(url_ema)

    desc_list.append('Time Series (15min)')
    desc_list.append('Time Series (15min)')
    desc_list.append('Technical Analysis: RSI')
    desc_list.append('Technical Analysis: STOCH')
    desc_list.append('Technical Analysis: STOCH')
    desc_list.append('Technical Analysis: SMA')
    desc_list.append('Technical Analysis: EMA')

    data_point_desc_list.append('4. close')
    data_point_desc_list.append('5. volume')
    data_point_desc_list.append('RSI')
    data_point_desc_list.append('SlowK')
    data_point_desc_list.append('SlowD')
    data_point_desc_list.append('SMA')
    data_point_desc_list.append('EMA')

    should_standardize.append(False)
    should_standardize.append(True)
    should_standardize.append(True)
    should_standardize.append(True)
    should_standardize.append(True)
    should_standardize.append(True)
    should_standardize.append(True)

    data_points = retrieve_features(urls, desc_list, data_point_desc_list, should_standardize)
    X, y = get_features_np(data_points)

    return X,y

X,y = retrieve_data()
print(X)
print(y)
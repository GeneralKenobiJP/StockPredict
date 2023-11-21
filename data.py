### Imports
from os import path
from os import mkdir

import time
import datetime
from dateutil.relativedelta import relativedelta

import requests
import numpy as np
from sklearn import preprocessing
import json

import config

### Alpha Vantage API handling
api_key = config.api_key
NUMBER_OF_FEATURES = 7

### ### ### CUSTOM FUNCTIONS
def create_url(function, symbol, interval, month, time_period=None, series_type=None, output_size=None):
    """
    Creates an url query string for Alpha Vantage API, using provided parameters
    :param function: API function (TIME_SERIES_INTRADAY for example)
    :param symbol: Stock ticker (GOOGL, for example)
    :param interval: Chosen time interval for date (15min for example)
    :param month: Which month do we want our data from
    :param time_period: Time period used for calculation of certain indicators (integer)
    OPTIONAL
    :param series_type: close/open/high/low (typically close)
    OPTIONAL
    :param output_size: 'compact' or 'full. Not used by technical indicators
    OPTIONAL
    :return: URL for Alpha Vantage API
    """
    if output_size is not None:
        return f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&interval={interval}&month={month}&outputsize={output_size}&apikey={api_key}'
    if time_period is None and series_type is None:
        return f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&interval={interval}&month={month}&apikey={api_key}'
    if series_type is None:
        return f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&interval={interval}&month={month}&time_period={time_period}&apikey={api_key}'
    if time_period is None:
        return f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&interval={interval}&month={month}&series_type={series_type}&apikey={api_key}'

    return f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&interval={interval}&month={month}&time_period={time_period}&series_type={series_type}&apikey={api_key}'

def get_urls(symbol, interval, month):
    """
    Retrieves urls that we usually want our model to get
    THIS FUNCTION WILL BE MODIFIED BASED ON OUR CURRENT DATA NEEDS
    FOR THE MODEL
    :param symbol: Stock ticker (GOOGL for example)
    :param interval: Chosen time interval for data (15min for example)
    :param month: Which month do we want our data from
    :return: List of urls for Alpha Vantage API
    """
    urls = []
    urls.append(create_url('TIME_SERIES_INTRADAY', symbol, interval, month, series_type='close', output_size='full'))
    urls.append(create_url('TIME_SERIES_INTRADAY', symbol, interval, month, series_type='close', output_size='full'))
    urls.append(create_url('RSI', symbol, interval, month, time_period=40, series_type='close'))
    urls.append(create_url('STOCH', symbol, interval, month, time_period=40))
    urls.append(create_url('STOCH', symbol, interval, month, time_period=40))
    urls.append(create_url('SMA', symbol, interval, month, time_period=40, series_type='close'))
    urls.append(create_url('EMA', symbol, interval, month, time_period=40, series_type='close'))

    return urls

def get_urls_time_range(symbol, interval, first_month, last_month):
    """
    Retrieves urls that we usually want our model to get
    in a specific time frame
    :param symbol: Stock ticker (GOOGL for example)
    :param interval: Chosen time interval for data (15min for example)
    :param first_month: Fist month we want our urls to refer to (YYYY-MM)
    :param last_month: Last month we want our urls to refer to (YYYY-MM)
    :return: List of lists of urls for Alpha Vantage API
    [time][feature]
    """
    urls = []
    first_date_str = first_month.split('-')
    last_date_str = last_month.split('-')
    first_date = datetime.datetime(int(first_date_str[0]),int(first_date_str[1]),1)
    last_date = datetime.datetime(int(last_date_str[0]), int(last_date_str[1]),1)
    timestamp = first_date
    while timestamp <= last_date:
        current_month_string = str(timestamp.year)+'-'+str(timestamp.month//10)+str(timestamp.month%10)
        urls.append(get_urls(symbol, interval, current_month_string))
        timestamp = timestamp + relativedelta(months=1)

    return urls

def parse_filename(filename):
    """
    Given an initial intended filename, the function parses it for
    forbidden chars (":") and converts to "-"
    :param filename: Initial intended filename (string)
    :return: Correct filename
    """
    return filename.replace(":","-").replace(" ","_")
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

def fetch_data_from_file(file_path):
    """
    Fetches data from a selected file
    :param file_path: selected file
    :return: JSON data
    """
    with open(file_path, 'r') as json_file:
        return json.load(json_file)

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

def retrieve_features(symbol, url_list, desc_list, data_point_desc_list, first_month, should_standardize,
                      use_filedata=False, save_data=True):
    """
    Given a description of data and urls, it retrieves the list of
    data points we are interested in
    :param symbol: Stock ticker (GOOGL for example)
    :param url_list: list of lists of Alpha Vintage API urls for our data
    [time][feature]
    :param desc_list: list of names of databases
    :param data_point_desc_list: list of names of data points
    :param first_month: First month we want our data from
    :param should_standardize: list of booleans for whether data should
    be standardized or not
    :param use_filedata: If true, it loads data from a file
    (it had previously been imported from Alpha Vantage).
    Otherwise, it loads data from Alpha Vantage API
    False, by default.
    :param save_data: If true, it saves data to a file.
    A prerequisite for this is to have use_filedata=False.
    True, be default.
    :return: list of data points
    """
    FILE_PATH = "data/"+str(symbol)+"/"
    WRITE_MODE = "w"
    retrieved_data_points = []
    current_response = None

    first_date_str = first_month.split('-')
    first_date = datetime.datetime(int(first_date_str[0]), int(first_date_str[1]), 1)
    timestamp = first_date
    current_month_string = first_month

    for j in range(url_list.__len__()):
        retrieved_data = []
        for i in range(url_list[j].__len__()):
            if i == 0 or not url_list[j][i].__eq__(url_list[j][i-1]):
                if not use_filedata:
                    current_response = fetch_data(url_list[j][i])
                else:
                    current_response = fetch_data_from_file(parse_filename(FILE_PATH+desc_list[i]+'/'+current_month_string))

                if save_data:
                    if use_filedata:
                        raise Exception("use_filedata and save_data are mutually exclusive")
                    this_file_path = parse_filename(FILE_PATH+desc_list[i])
                    if not path.isdir(FILE_PATH):
                        mkdir(FILE_PATH)
                    if not path.isdir(this_file_path):
                        mkdir(this_file_path)
                    with open(parse_filename(this_file_path+'/'+current_month_string), WRITE_MODE) as json_file:
                        json.dump(current_response, json_file)

            retrieved_data.append(current_response[desc_list[i]])

            if j!=0:
                retrieved_data_points[i].extend(
                    [float(data_point[data_point_desc_list[i]])
                     for data_point in retrieved_data[i].values()]
                )
            else:
                retrieved_data_points.append(
                    [float(data_point[data_point_desc_list[i]])
                    for data_point in retrieved_data[i].values()]
                )
            if should_standardize[i]:
                retrieved_data_points[i] = standardize_list(retrieved_data_points[i])


        timestamp = timestamp + relativedelta(months=1)
        current_month_string = str(timestamp.year) + '-' + str(timestamp.month // 10) + str(timestamp.month%10)

    return retrieved_data_points

def get_number_of_features():
    """
    Get number of features we retrieve from data
    :return: number of features for the model
    """
    return NUMBER_OF_FEATURES

def retrieve_data(symbol, interval='15min', first_month='2023-10', last_month='2023-10', use_filedata=False, save_data=True):
    """
    Main API function in data.py
    Retrieve preselected data to be fed into a model
    Feature types:
    Closing prices, volumes, RSI, Stoch, SMA, EMA
    :param symbol: Stock ticker (GOOGL for example)
    :param interval: Chosen time interval for data (15min for example)
    :param first_month: First month we want our data from
    :param last_month: Last month we want our data from
    :param use_filedata: If true, it loads data from a file
    (it had previously been imported from Alpha Vantage).
    Otherwise, it loads data from Alpha Vantage API
    False, by default.
    :param save_data: If true, it saves data to a file.
    A prerequisite for this is to have use_filedata=False.
    True, be default.
    :return: model data X,y (numpy arrays).
    Processed and ready to be fed into a model
    """
    #urls = []
    desc_list = []
    data_point_desc_list = []
    should_standardize = []

    urls = get_urls_time_range(symbol, interval, first_month, last_month)

    #urls.append(url_prices)
    #urls.append(url_prices)
    #urls.append(url_rsi)
    #urls.append(url_stoch)
    #urls.append(url_stoch)
    #urls.append(url_sma)
    #urls.append(url_ema)

    #urls = get_urls(symbol, interval, month)

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

    data_points = retrieve_features(symbol, urls, desc_list, data_point_desc_list, first_month, should_standardize, use_filedata, save_data)
    X, y = get_features_np(data_points)

    return X,y

X,y = retrieve_data('MSFT', '15min',first_month='2023-08' ,last_month='2023-10' , use_filedata=False, save_data=True)
#X,y = retrieve_data(True, False)
print(X)
print(y)

#{'Information': 'Thank you for using Alpha Vantage! Our standard API rate limit is 25 requests per day. Please subscribe to any of the premium plans at https://www.alphavantage.co/premium/ to instantly remove all daily rate limits.'}
#Should test for that
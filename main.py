import numpy as np
import requests
import tensorflow as tf
import matplotlib.pyplot as plt

# Alpha Vantage API Key and endpoint
import config
api_key = config.api_key
symbol = 'GOOGL'
interval = '15min'  # Adjust the interval as needed
outputsize = 'full'
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={api_key}'
time_period = '40'
series_type = 'close'
url_rsi = f'https://www.alphavantage.co/query?function=RSI&symbol={symbol}&interval={interval}&time_period={time_period}&series_type={series_type}&apikey={api_key}'
url_stoch = f'https://www.alphavantage.co/query?function=STOCH&symbol={symbol}&interval={interval}&apikey={api_key}'

EPOCH_NUM = 50

# Fetch intraday stock data
response = requests.get(url)
data = response.json()
intraday_data = data['Time Series (15min)']
response = requests.get(url_rsi)
data = response.json()
rsi_data = data['Technical Analysis: RSI']
response = requests.get(url_stoch)
data = response.json()
stoch_data = data['Technical Analysis: STOCH']

# Extract closing prices
closing_prices = [float(data_point['4. close']) for data_point in intraday_data.values()]
volumes = [float(data_point['5. volume']) for data_point in intraday_data.values()]
rsi_indexes = [float(data_point['RSI']) for data_point in rsi_data.values()]
stoch_indexes_k = [float(data_point['SlowK']) for data_point in stoch_data.values()]
stoch_indexes_d = [float(data_point['SlowD']) for data_point in stoch_data.values()]

print(closing_prices.__len__())
print(volumes.__len__())
print(rsi_indexes.__len__())
print(stoch_indexes_k.__len__())
print(stoch_indexes_d.__len__())

# Prepare data for supervised learning
X = []
y = []
window_size = 20  # Adjust the window size as needed
num_feature_classes = 5

length = min(len(closing_prices), len(rsi_indexes), len(stoch_indexes_d), len(stoch_indexes_k))

for i in range(length - window_size):
    feature_vector = []

    # Append closing prices
    feature_vector.append(closing_prices[i:i + window_size])

    # Append volumes
    feature_vector.append(volumes[i:i + window_size])

    # Append RSI values
    feature_vector.append(rsi_indexes[i:i + window_size])

    # Append Stochastic Oscillator values (K and D)
    feature_vector.append(stoch_indexes_k[i:i + window_size])
    feature_vector.append(stoch_indexes_d[i:i + window_size])

    #
    X.append(feature_vector)
    #for j in range(num_feature_classes):
    #    X[j] = np.array()
    y.append(closing_prices[i+window_size])

X = np.array(X)
y = np.array(y)

# Split data into training and testing sets
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

models = []

# Create a simple neural network model
default_model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(num_feature_classes,window_size)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1)  # Single output neuron for price prediction
])

# Compile the model
custom_learning_rate = 0.00001  # You can adjust this value
custom_optimizer = tf.keras.optimizers.Adam(learning_rate=custom_learning_rate)
default_model.compile(optimizer=custom_optimizer, loss='mean_squared_error')

# Train the model
model = default_model
model.fit(X_train, y_train, epochs=EPOCH_NUM, batch_size=64, validation_data=(X_test, y_test))

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model (you can use various metrics to evaluate the model)
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# You can use the trained model to make future predictions as well

x_time = np.arange(len(X_train[0,:]))

#print(predictions)

diff = predictions[:,0] - y_test
plt.plot(diff, c='g', label='Delta')

#plt.plot(predictions,c='b',label='Pred')

#plt.plot(X_train[0,:],c='r',label='actual')
#plt.scatter(x_time,X_train[0,:], window_size ,marker='x',c='r',label='actual')
plt.title('GOOGL')
plt.ylabel('Prediction error')
plt.xlabel('Time')
plt.legend()
plt.show()

plt.plot(predictions,c='b',label='Pred')

#plt.plot(X_train[0,:],c='r',label='actual')
plt.plot(y_test,c='r',label='actual')
plt.title('GOOGL')
plt.ylabel('Price')
plt.xlabel('Time')
plt.legend()
plt.show()

#print(predictions)
'''
print(X.shape)
print(y.shape)
print(predictions.shape)
print(y_test.shape)
print("\n")
print("\n")
print("X: ")
print(X)
print("y: ")
print(y)
print("Predictions: ")
print(predictions)
'''

print(X.shape)
print(X)


### model.save('my_model.h5')
### loaded_model = keras.models.load_model('my_model.h5')
### ### ### INTRODUCTION
### Imports
import numpy as np
import requests
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing
import config

import data

X,y = data.retrieve_data('GOOGL', '15min', first_month='2023-02', last_month='2023-10', use_filedata=True, save_data=False)
num_feature_classes = data.get_number_of_features()

### CONSTANTS
EPOCH_NUM = 25
custom_learning_rate = 0.02
window_size = 20

### Split data into training and testing sets
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

### ### ### MODEL

# Create a simple neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(num_feature_classes*8, activation='relu', input_shape=(num_feature_classes,window_size)),
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    #tf.keras.layers.LSTM(num_feature_classes,activation='tanh',recurrent_activation='sigmoid',use_bias=True),
    tf.keras.layers.Dense(num_feature_classes, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1)  # Single output neuron for price prediction
])
"""model = tf.keras.Sequential([
    tf.keras.layers.LSTM(num_feature_classes, activation='tanh', recurrent_activation='sigmoid',use_bias=True)
])"""

# Compile the model
custom_optimizer = tf.keras.optimizers.Adam(learning_rate=custom_learning_rate)
model.compile(optimizer=custom_optimizer, loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=EPOCH_NUM, batch_size=64, validation_data=(X_test, y_test))

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

#print(predictions)
### PLOTS
diff = predictions[:,0] - y_test
plt.plot(diff, c='g', label='Delta')
plt.title('GOOGL')
plt.ylabel('Prediction error')
plt.xlabel('Time')
plt.legend()
plt.show()

plt.plot(predictions,c='b',label='Pred')
plt.plot(y_test,c='r',label='actual')
plt.title('GOOGL')
plt.ylabel('Price')
plt.xlabel('Time')
plt.legend()
plt.show()

###
### DEBUG SECTION
###
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

#print(X.shape)
#print(X)
#print (model.get_weights())


### model.save('my_model.h5')
### loaded_model = keras.models.load_model('my_model.h5')
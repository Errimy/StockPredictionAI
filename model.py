import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
# Generate synthetic data for demonstration purposes
data = np.random.random((1000, 10))
target = np.random.random((1000, 1))
# Build the model
model = Sequential()
model.add(LSTM(32, input_shape=(10, 1)))
model.add(Dense(1))
model.compile(optimizer='adam',
loss='mean_squared_error')
# Train the model
model.fit(data, target, epochs=10)
# Generate test data
X_test = np.random.random((100, 10, 1))
y_test = np.random.random((100, 1))

# Make predictions on the test data
predicted_prices = model.predict(X_test)

# Plot the predicted prices and actual prices
plt.plot(predicted_prices, 'r', label='Predicted Prices')
plt.plot(y_test, 'b', label='Actual Prices')
plt.xlabel('Sample')
plt.ylabel('Price')
plt.title('Predicted prices vs Actual prices')
plt.legend()
plt.show()

tf.keras.models.save_model(model, 'StockPrediction.h5')
print('Model saved successfully !')


import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from keras.models import load_model
import matplotlib.pyplot as plt
from datetime import datetime

st.title("Stock Price Predictor")

stock = st.text_input("Enter the Stock name", "GOOG")

end_time = datetime.now()
start_time = datetime(end_time.year - 20, end_time.month, end_time.day)

stock_data = yf.download(stock, start = start_time, end = end_time, auto_adjust = False)

model = load_model("goog_stock_price_model.keras")

st.subheader("Stocks")
st.write(stock_data)

def graph(figsize,moving_average, original_data) :
    fig = plt.figure(figsize = figsize)
    plt.plot(moving_average, 'b')
    plt.plot(original_data['Adj Close'], 'y')
    plt.xlabel('years')
    plt.ylabel('Adjusted Close Price')

    return fig

st.subheader("Moving Average with 250 days window over Original Data")
stock_data['moving_average_with_250_days'] = stock_data['Adj Close'].rolling(250).mean()
st.pyplot(graph((14, 6), stock_data['moving_average_with_250_days'], stock_data))

st.subheader("Moving Average with 100 days window over Original Data")
stock_data['moving_average_with_100_days'] = stock_data['Adj Close'].rolling(100).mean()
st.pyplot(graph((14, 6), stock_data['moving_average_with_100_days'], stock_data))

splitting_len = int(0.8*len(stock_data))
X_test = pd.DataFrame(stock_data['Adj Close'][splitting_len:])

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0, 1))
scaled_data = scaler.fit_transform(X_test)

def create_sequences(data, window) :
    x = []
    y = []
    for i in range(window, len(scaled_data)) :
        x.append(data[i-window:i])
        y.append(data[i])
    return np.array(x), np.array(y)

X_data, y_data = create_sequences(scaled_data, 100)

predictions = model.predict(X_data)

inverse_predictions = scaler.inverse_transform(predictions)
inverse_test_output = scaler.inverse_transform(y_data)

output_data = pd.DataFrame(
    {
        'Original Test Data Price' : inverse_test_output.reshape(-1),
        'Predicted Price' : inverse_predictions.reshape(-1)
    },
    index = stock_data.index[splitting_len+100:]
)

st.subheader("Original Adjusted Close Price vs Predicted Price")
st.write(output_data)

st.subheader("Original Adjusted Close Price vs Predicted Price")
fig = plt.figure(figsize = (14, 6))
plt.plot(pd.concat([stock_data['Adj Close'][:splitting_len+100], output_data], axis = 0))
plt.legend(["Out of Window Data", "Original Data", "Predicted Data"])
st.pyplot(fig)
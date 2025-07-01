# 📈 Stock Price Predictor App

**Live Demo**: [https://stockpricepredictorsajalgarg.streamlit.app/](https://stockpricepredictorsajalgarg.streamlit.app/)

This is a Streamlit web application that predicts stock prices using an LSTM (Long Short-Term Memory) neural network model. It fetches 20 years of historical stock data via `yFinance`, visualizes moving averages, and compares actual vs predicted stock prices using a pre-trained LSTM model.

---

## 🔍 Features

- 📊 Fetch historical stock data for any stock (e.g., GOOGL, AAPL, TSLA)
- 📈 Visualize 100-day and 250-day moving averages
- 🤖 Load and use an LSTM deep learning model to predict prices
- 🔍 Compare actual vs predicted prices in graph and table format
- 📉 Scales data with MinMaxScaler for better model performance

---

## 🚀 Live Demo

👉 Click here: [https://stockpricepredictorsajalgarg.streamlit.app/](https://stockpricepredictorsajalgarg.streamlit.app/)

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/)
- [yFinance](https://pypi.org/project/yfinance/)
- [TensorFlow / Keras](https://www.tensorflow.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)

---

## 📁 Project Structure
📦 stock-price-predictor/
├── stock_price_predictor.py # Main Streamlit app
├── goog_stock_price_model.keras # Pre-trained LSTM model
---

## 📦 Installation & Running Locally

1. **Clone the repository**

    ```bash
    git clone https://github.com/sajal2206334/NSE_stock_price_predictor.git
    cd stock_price_predictor.py
    ```

2. **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the app**

    ```bash
    streamlit run app.py
    ```

---

## 🧠 How It Works

- Fetches long-term stock data using `yfinance`
- Calculates 100-day and 250-day moving averages
- Prepares sequences for prediction using a window size of 100
- Feeds the sequences to a trained LSTM model loaded from `.keras` file
- Compares predicted stock prices with actual values

---

## 🧩 Future Improvements

- [ ] Add anomaly detection system
- [ ] Implement auto-trade feature using Alpaca API
- [ ] Display confidence intervals for predictions
- [ ] Enable model upload for multiple stock predictions
- [ ] Add portfolio dashboard view

---

---

## 👨‍💻 Author

**Sajal Garg** 
Feel free to connect or reach out for feedback or collaboration!

import os
import pandas as pd
import numpy as np
import yfinance as yf
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the pre-trained models
models = {
    'AAPL': load_model('models/AAPL_lstm_model.h5'),
    'GOOGL': load_model('models/GOOGL_lstm_model.h5'),
    'MSFT': load_model('models/MSFT_lstm_model.h5')
}

# Function to fetch the latest stock price data
def fetch_latest_data(stock_symbol, days=60):
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)
    try:
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date, interval='1d')
        if stock_data.empty:
            raise ValueError("No data fetched.")
        return stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        print(f"Error fetching data for {stock_symbol}: {e}")
        return None

# Function to prepare data for prediction
def prepare_data(data, time_step=60):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    X = []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, :])
    X = np.array(X)
    return X, scaler

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    stock_symbol = request.form['stock_symbol']
    if stock_symbol not in models:
        return jsonify({'error': 'Model not found for the given stock symbol.'})
    
    model = models[stock_symbol]
    data = fetch_latest_data(stock_symbol)
    
    if data is None or data.empty:
        return jsonify({'error': 'Failed to fetch stock data.'})

    X, scaler = prepare_data(data)
    X = np.reshape(X, (X.shape[0], X.shape[1], data.shape[1]))
    
    # Predict the next 5 days
    predictions = []
    last_data = X[-1]
    for _ in range(5):
        prediction = model.predict(last_data.reshape(1, last_data.shape[0], last_data.shape[1]))
        predictions.append(prediction[0][0])
        new_data = np.append(last_data[1:], prediction, axis=0)
        last_data = new_data
    
    predictions = scaler.inverse_transform(np.concatenate([np.zeros((5, X.shape[2] - 1)), np.array(predictions).reshape(-1, 1)], axis=1))[:, -1]
    
    current_price = data['Close'].values[-1]
    
    return jsonify({
        'stock_symbol': stock_symbol,
        'current_price': current_price,
        'predicted_prices': predictions.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
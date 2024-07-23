import os
import pandas as pd
import numpy as np
import yfinance as yf
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import datetime  # Make sure to import datetime

app = Flask(__name__)

# Load the pre-trained models
models = {
    'AAPL': load_model('models/AAPL_lstm_model.h5'),
    'GOOGL': load_model('models/GOOGL_lstm_model.h5'),
    'MSFT': load_model('models/MSFT_lstm_model.h5')
}

def fetch_latest_data(stock_symbol, days=120):
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

    # Ensure there is at least one time step available
    if X.size == 0:
        return None, None

    return X, scaler

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    request_data = request.get_json()
    stock_symbol = request_data.get('stock_symbol', '')
    
    if stock_symbol not in models:
        return jsonify({'error': 'Model not found for the given stock symbol.'})
    
    model = models[stock_symbol]
    data = fetch_latest_data(stock_symbol)
    
    if data is None or data.empty:
        return jsonify({'error': 'Failed to fetch stock data.'})

    # Check if we have enough data points
    if len(data) < 60:
        return jsonify({'error': f'Not enough data to make predictions. Only {len(data)} data points available.'})

    X, scaler = prepare_data(data)

    if X is None:
        return jsonify({'error': 'Not enough data to make predictions.'})

    X = np.reshape(X, (X.shape[0], X.shape[1], data.shape[1]))

    # Predict the next 5 days
    predictions = []
    last_data = X[-1]  # Shape: (60, 5)
    for _ in range(5):
        prediction = model.predict(last_data.reshape(1, last_data.shape[0], last_data.shape[1]))  # Shape: (1, 1)
        # Add padding to match the input shape except for the last column
        prediction_padded = np.zeros((last_data.shape[1],))
        prediction_padded[-1] = prediction[0][0]
        
        # Concatenate the prediction with the last_data
        new_data = np.append(last_data[1:], [prediction_padded], axis=0)
        predictions.append(prediction[0][0])
        last_data = new_data
    
    # Convert predictions back to the original scale
    predictions_scaled = np.array(predictions).reshape(-1, 1)
    predictions_padded = np.concatenate(
        [np.zeros((predictions_scaled.shape[0], X.shape[2] - 1)), predictions_scaled], axis=1
    )
    predictions = scaler.inverse_transform(predictions_padded)[:, -1]
    
    # Extract current prices for plotting
    current_prices = data['Close'].values[-60:].tolist()  # Take the last 60 days for plotting

    return jsonify({
        'stock_symbol': stock_symbol,
        'current_prices': current_prices,
        'predicted_prices': predictions.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
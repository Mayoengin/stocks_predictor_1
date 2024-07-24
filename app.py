import yfinance as yf
import numpy as np
import pandas as pd
from keras.models import load_model
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib

# Define the custom metric function
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true))

# Register the custom metric function
tf.keras.utils.get_custom_objects().update({'mse': mse})

# Initialize Flask app
app = Flask(__name__)

# Load the models and scalers with custom objects
models = {
    'AAPL': load_model('models/AAPL_lstm_model.h5', custom_objects={'mse': mse}),
    'GOOGL': load_model('models/GOOGL_lstm_model.h5', custom_objects={'mse': mse}),
    'MSFT': load_model('models/MSFT_lstm_model.h5', custom_objects={'mse': mse}),
}

scalers = {
    'AAPL': joblib.load('models/AAPL_scaler.pkl'),
    'GOOGL': joblib.load('models/GOOGL_scaler.pkl'),
    'MSFT': joblib.load('models/MSFT_scaler.pkl'),
}

def fetch_stock_data(ticker):
    try:
        stock_data = yf.download(ticker, period='1mo', interval='1d')
        stock_data = stock_data['Close'].values
        return stock_data
    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")
        return None

def preprocess_data(data, scaler):
    try:
        print(f"Raw data: {data}")
        data = np.array(data).reshape(-1, 1)  # Ensure data is a 2D array
        scaled_data = scaler.transform(data)
        print(f"Scaled data: {scaled_data}")
        data = scaled_data.reshape((1, scaled_data.shape[0], 1))
        return data
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        request_data = request.get_json()
        stock = request_data['stock']
    except KeyError:
        return jsonify({'error': 'No stock ticker provided'}), 400
    
    model = models.get(stock)
    scaler = scalers.get(stock)
    
    if not model or not scaler:
        return jsonify({'error': 'Model or scaler not found'}), 404
    
    stock_data = fetch_stock_data(stock)
    if stock_data is None:
        return jsonify({'error': 'Error fetching stock data'}), 500
    
    preprocessed_data = preprocess_data(stock_data[-3:], scaler)  # Use last 3 days of closing prices
    if preprocessed_data is None:
        return jsonify({'error': 'Error preprocessing data'}), 500
    
    try:
        print(f"Scaler data_min_ for {stock}: {scaler.data_min_}")
        print(f"Scaler data_max_ for {stock}: {scaler.data_max_}")
        print(f"Preprocessed data for {stock}: {preprocessed_data}")
        prediction = model.predict(preprocessed_data)
        print(f"Raw prediction for {stock}: {prediction}")
        
        # Inverse transform the prediction
        next_day_price = scaler.inverse_transform(prediction)[0, 0]  # Convert back to original scale
        next_day_price = float(next_day_price)  # Convert to standard Python float

        # Print the current prices and the predicted price
        print(f"Current prices for {stock}: {stock_data.tolist()}")
        print(f"Predicted next day price for {stock}: {next_day_price}")

        current_prices = stock_data.tolist()
        predicted_prices = [next_day_price]
        return jsonify({'current_prices': current_prices, 'predicted_prices': predicted_prices})
    except Exception as e:
        print(f"Error making prediction: {e}")
        return jsonify({'error': 'Error making prediction'}), 500

if __name__ == '__main__':
    app.run(debug=True)

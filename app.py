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
    stock_data = yf.download(stock_symbol, period='60d', interval='1d')
    return stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]

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
    
    X, scaler = prepare_data(data)
    X = np.reshape(X, (X.shape[0], X.shape[1], data.shape[1]))
    
    prediction = model.predict(X[-1].reshape(1, X.shape[1], X.shape[2]))
    prediction = scaler.inverse_transform(np.concatenate([np.zeros((1, X.shape[2] - 1)), prediction], axis=1))[:, -1]
    
    current_price = data['Close'].values[-1]
    
    return jsonify({
        'stock_symbol': stock_symbol,
        'current_price': current_price,
        'predicted_price': prediction[0]
    })

@app.route('/upload', methods=['POST'])
def upload():
    if 'model' not in request.files:
        return jsonify({'error': 'No file part in the request.'})
    
    file = request.files['model']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading.'})
    
    stock_symbol = request.form['stock_symbol']
    if stock_symbol not in models:
        return jsonify({'error': 'Invalid stock symbol.'})
    
    file.save(os.path.join('', f'{stock_symbol}_lstm_model.h5'))
    models[stock_symbol] = load_model(f'{stock_symbol}_lstm_model.h5')
    
    return jsonify({'message': f'Model for {stock_symbol} uploaded successfully.'})

if __name__ == '__main__':
    app.run(debug=True)

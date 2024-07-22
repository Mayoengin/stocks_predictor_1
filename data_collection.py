import yfinance as yf
import pandas as pd
import os

def fetch_stock_data(stock_symbol, start_date, end_date, data_dir='data'):
    """
    Fetches historical stock data for a given symbol and saves it as a CSV file.

    Parameters:
        stock_symbol (str): The stock symbol to fetch data for.
        start_date (str): The start date for fetching data (format: 'YYYY-MM-DD').
        end_date (str): The end date for fetching data (format: 'YYYY-MM-DD').
        data_dir (str): The directory to save the fetched data.
    """
    # Fetch the stock data
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    
    # Ensure the data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Save the data to a CSV file
    file_path = os.path.join(data_dir, f'{stock_symbol}_data.csv')
    stock_data.to_csv(file_path)
    
    print(f"Data for {stock_symbol} saved to {file_path}")

if __name__ == "__main__":
    # Example usage
    stock_symbols = ['AAPL', 'GOOGL', 'MSFT']  # List of stock symbols to fetch
    start_date = '2020-01-01'
    end_date = '2023-07-01'

    for symbol in stock_symbols:
        fetch_stock_data(symbol, start_date, end_date)

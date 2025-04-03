import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import yfinance as yf
from datetime import datetime
import ta

# Function to fetch stock data
def get_stock_data(ticker="AAPL", start="2020-01-01"):
    end = datetime.today().strftime('%Y-%m-%d')
    stock = yf.download(ticker, start=start, end=end)
    return stock

def train_model(df):
    df['Target'] = df['Close'].shift(-1)  # Predict next day's Close price
    df.dropna(inplace=True)  # Remove NaN values
    print(df.isna().sum())  # Check if there are NaN values in the DataFrame
    print(df.tail())

    X = df[['Open', 'High', 'Low', 'Close', 'Volume']]  # Features
    y = df['Target']  # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test
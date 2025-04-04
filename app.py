import streamlit as st
import pandas as pd
import numpy as np
from model import get_stock_data, train_model  # Import from your stock_model.py file
from sklearn.metrics import mean_absolute_error, mean_squared_error

# List of tickers with company names
tickers = [
    ("AAPL", "Apple Inc."),
    ("MSFT", "Microsoft Corp."),
    ("TSLA", "Tesla Inc."),
    ("GOOGL", "Alphabet Inc."),
    ("AMZN", "Amazon.com Inc."),
    ("META", "Meta Platforms Inc."),
    ("NVDA", "NVIDIA Corporation"),
    ("AMD", "Advanced Micro Devices"),
    ("NFLX", "Netflix Inc."),
    ("BABA", "Alibaba Group"),
    ("INTC", "Intel Corporation"),
    ("V", "Visa Inc."),
    ("MA", "Mastercard Inc."),
    ("DIS", "The Walt Disney Company"),
    ("JNJ", "Johnson & Johnson"),
    ("PG", "Procter & Gamble"),
    ("KO", "Coca-Cola Company"),
    ("WMT", "Walmart Inc."),
    ("PEP", "PepsiCo Inc."),
    ("GS", "Goldman Sachs"),
    ("JPM", "JPMorgan Chase & Co."),
    ("HD", "Home Depot Inc."),
    ("BA", "Boeing Company"),
    ("UNH", "UnitedHealth Group"),
    ("CVX", "Chevron Corporation"),
    ("XOM", "ExxonMobil Corporation"),
    ("RDS.A", "Royal Dutch Shell"),
    ("LMT", "Lockheed Martin"),
    ("T", "AT&T Inc."),
    ("VZ", "Verizon Communications Inc.")
]

# Streamlit User Interface
st.title("Stock Price Predictor")
st.markdown('<h3 style="font-size: 20px; font-weight: light;">Select a stock ticker to predict its future price.</h3>', unsafe_allow_html=True)


# Dropdown for selecting stock ticker
ticker = st.selectbox("Select Stock Ticker", tickers, format_func=lambda x: f"{x[0]} ({x[1]})")

st.divider()
# Fetch stock data
df = get_stock_data(ticker[0])  # Use the ticker symbol (first element of the tuple)

if df.empty:
    st.error(f"Could not fetch data for {ticker[0]}. Please try again.")
else:
    st.write(f"Data for {ticker[0]} from {df.tail(5).index.min()} to {df.tail(5).index.max()}")
    st.dataframe(df.tail())

    # Train the model
    model, X_test, y_test = train_model(df)
    
    # Show prediction
    latest_data = df.iloc[-1][['Open', 'High', 'Low', 'Close', 'Volume']].values.reshape(1, -1)
    predicted_price = model.predict(latest_data)
    
    st.markdown(f"<span style='font-size: 20px;'>Predicted Next Day's Closing Price for {ticker[0]}:  <b>${predicted_price[0]:.2f}</b></span>", unsafe_allow_html=True)

    st.divider()
    
    # Optional: Display performance metrics
    st.subheader("Model Performance")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    st.write(f"Mean Absolute Error: {mae:.2f}")
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"Root Mean Squared Error: {np.sqrt(mse):.2f}")

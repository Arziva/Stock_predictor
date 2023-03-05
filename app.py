import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pandas_datareader as data #scrape
from keras.models import load_model
import yfinance as yf
yf.pdr_override()
import streamlit as st

import datetime
start = '2009-01-01'
end = '2020-12-31'

#df = data.DataReader('AAPL','yahoo', start, end)
#df.head()

#stocks = ["stock1","stock2", ...]
st.title('Stock Trend Predictor')

user_input = st.text_input('Enter stock Ticker', 'ASHOKLEY.NS')

start = datetime.datetime(2009,1,1)
end = datetime.datetime(2023,1,1)
df = yf.download(user_input, start=start, end=end, interval='1d', progress=False)["Adj Close"]

#describe data
st.subheader("Data from 2010 - 2019")
st.write(df.describe())


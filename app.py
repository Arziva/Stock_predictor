import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pandas_datareader as data #scrape
import tensorflow as tf
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

user_input = st.text_input('Enter stock Ticker', 'AAPL')

start = datetime.datetime(2009,1,1)
end = datetime.datetime(2023,1,1)
df = yf.download(user_input, start=start, end=end)

#describe data
st.subheader("Data from 2010 - 2019")
st.write(df.describe())

#visualizations
st.subheader('Closing Price vs Time Chart')
fig = plt.figure( figsize= (12,6) )
plt.plot(df.Close)
st.pyplot(fig)




st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure( figsize= (12,6) )
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)



st.subheader('Closing Price vs Time Chart with 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure( figsize= (12,6) )
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)


#train_test seperation
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.75)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.75):int(len(df))])
print(data_training.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler  = MinMaxScaler(feature_range = (0,1))

data_training_array = scaler.fit_transform(data_training)


#loading LSTM model bruh.. Arun.. try not to sleep..
model = tf.keras.models.load_model('stock_arun.h5')
#model = load_model('stock_arun.h5')



#testing here..

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i, 0])

x_test = np.array(x_test)
y_test = np.array(y_test)

#making predictions
y_predicted = model.predict(x_test)
scaler = scaler.scale_

#upscaling the values back to normal
scale_factor = 1/scaler[0] 
y_predicted *= scale_factor
y_test *= scale_factor

#Final Graph
st.subheader("Predictions vs Original")
fig2 = plt.figure(figsize = (12, 6))
plt.plot(y_test, 'b', label = 'Original price')
plt.plot(y_predicted, 'r', label = 'Predicted by model')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


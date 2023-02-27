
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data #scrape
import yfinance as yf
yf.pdr_override()

import datetime
start = '2009-01-01'
end = '2020-12-31'

#df = data.DataReader('AAPL','yahoo', start, end)
#df.head()

#stocks = ["stock1","stock2", ...]
start = datetime.datetime(2009,1,1)
end = datetime.datetime(2023,1,1)
df = yf.download('ASHOKLEY.NS', start=start, end=end)
df.head()

df.tail()

df = df.reset_index()
df.head()

df = df.drop(['Date'], axis = 1)
df.head()

plt.plot(df.Close)

df

moving_avg100 = df.Close.rolling(100).mean()
moving_avg100

plt.figure(figsize = (12,6))
plt.plot(df.Close, 'b')
plt.plot(moving_avg100,'r')

moving_avg200 = df.Close.rolling(200).mean()
moving_avg200

plt.figure(figsize = (12,6))
plt.plot(df.Close, 'b')
plt.plot(moving_avg100,'r')
plt.plot(moving_avg200, 'g')

df.shape

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.75)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.75):int(len(df))])

print(data_training.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler  = MinMaxScaler(feature_range = (0,1))

data_training_array = scaler.fit_transform(data_training)
data_training_array

x_train = [] #stock of first 100 days
y_train = [] #predicted stock of 101th day

#it moves so on and so forth

for i in range(100, data_training_array.shape[0]):
  x_train.append(data_training_array[i-100:i]) #from 0-100. like that
  y_train.append(data_training_array[i,0]) #the next value in the coloumn

x_train, y_train = np.array(x_train), np.array(y_train)

x_train.shape

from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

model = Sequential()
model.add(LSTM(units = 50, activation = 'relu', return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
model.add(Dropout(0.3))

model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
model.add(Dropout(0.4))

model.add(LSTM(units = 120, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(units = 1))

model.summary()


model.compile(optimizer = 'adam', loss = 'mean_squared_error') #msr good for time series sequential data
model.fit(x_train, y_train, epochs = 50)

model.save('stock_arun.h5')

data_testing.head()

data_training.tail(100)

past_100_days = data_training.tail(100)

final_df = past_100_days.append(data_testing, ignore_index = True)

final_df.head()

input_data = scaler.fit_transform(final_df)
input_data

input_data.shape

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i, 0])

x_test = np.array(x_test)
y_test = np.array(y_test)
print(x_test.shape)
print(y_test.shape)

#predicting

y_predicted = model.predict(x_test)


y_predicted.shape

scaler.scale_

scale_factor = 1/0.00757002
y_predicted *= scale_factor
y_test *= scale_factor

plt.figure(figsize = (12, 6))
plt.plot(y_test, 'b', label = 'Original price')
plt.plot(y_predicted, 'r', label = 'Predicted by model')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import requests
from data_get import load_data_1, load_data_2  # do like this

# csv_url = "https://query1.finance.yahoo.com/v7/finance/download/AAPL?period1=1583106669&period2=1614642669&interval=1d&events=history&includeAdjustedClose=true"
# print(csv_url)
# req = requests.get(csv_url)
# url_content = req.content
# csv_file = open('AAPL.csv', 'wb')
# csv_file.write(url_content)
# csv_file.close()
# dataset = pd.read_csv('AAPL.csv')
# print(dataset)
df = load_data_2()  # there are some coins with NaNs in old dates
# plotting the graph
idx = pd.IndexSlice
df = df.loc[:, idx['BTC-USD', :]]  # coin
df.columns = df.columns.droplevel(0)
df = df.drop(['Adj Close','Open','High','Low','Volume'], axis=1)
#df1 = preprocess_1(df1)
print(df)
print(df['Close'].values.shape)

# dataset['Date'] = pd.to_datetime(dataset.Date ,format='%Y-%m-%d')
# dataset.index = dataset['Date']
# plt.figure(figsize=(16,8))
# plt.plot(dataset['Volume'], label='Amount of Stocks')
#
# dataset = dataset.sort_index(ascending=True, axis=0)
# dataset2 = pd.DataFrame(index=range(0 ,len(dataset)) ,columns=['Date', 'Close'])
# for i in range(0 ,len(dataset)):
#     dataset2['Date'][i] = dataset['Date'][i]
#     dataset2['Close'][i] = dataset['Close'][i]
#
# dataset2.index = dataset2.Date
# dataset2.drop('Date', axis=1, inplace = True)

dataset3 = df.values

train = dataset3[0:400, :]
valid = dataset3[400:, :]
print(train)
print(valid)
print(dataset3.shape)


# puts everything between (0,1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset3)
#print(scaled_data)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

model = Sequential()
model.add(LSTM( units=100, return_sequences = True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=100))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=16, verbose=2)
print(model.inputs)
X_test = []
for i in range(60,len(model.inputs)): #for i in range(60,model.inputs.shape[0]):
    X_test.append(model.inputs[i-60:i,0])
X_test = np.array(X_test)
print(X_test)
print(X_test.shape)
X_test = np.reshape(X_test.shape[0],X_test.shape[1],1)
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)
#print(X_test.shape)

# train = dataset2[:200]
# valid = dataset2[200:]
train = df[:400]
valid = df[400:]
valid['Predictions'] = closing_price
plt.figure(figsize=(20,10))
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import time
from datetime import datetime, timedelta


def train_forecast1(df):
    df = df.dropna(axis=0)
    # fit the format of the scaler -> convert shape from (1000, ) -> (1000, 1)
    y = df['Close'].values.reshape(-1, 1)
    # scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(y)
    y = scaler.transform(y)

    # generate the input and output sequences
    n_lookback = 60  # length of input sequences (lookback period)
    n_forecast = 30  # length of output sequences (forecast period)

    X = []
    Y = []

    for i in range(n_lookback, len(y) - n_forecast + 1):
        X.append(y[i - n_lookback: i])
        Y.append(y[i: i + n_forecast])

    X = np.array(X)
    Y = np.array(Y)

    # fit the model
    model = Sequential()
    model.add(LSTM(units=25, return_sequences=True, input_shape=(n_lookback, 1)))
    model.add(Dropout(rate=0.1))
    model.add(keras.layers.Bidirectional(LSTM(units=50, return_sequences=True)))
    model.add(Dropout(rate=0.1))
    model.add(keras.layers.Bidirectional(LSTM(units=25, return_sequences=False)))
    model.add(Dense(units=n_forecast, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    print(model.summary())
    start = time.time()
    history = model.fit(X, Y, epochs=100, batch_size=32, verbose=0, validation_split=0.2) # 100 epochs
    stop = time.time()
    print(f"Training time: {(stop - start):.2f}s")
    #history = regressor.fit(X_train, y_train, validation_split=0.3, epochs=40, batch_size=64, callbacks=[es])
    print(history.history.keys())

    # generate the forecasts
    X_ = y[- n_lookback:]  # last available input sequence
    X_ = X_.reshape(1, n_lookback, 1)

    Y_ = model.predict(X_).reshape(-1, 1)
    Y_ = scaler.inverse_transform(Y_)

    # organize the results in a data frame
    df_past = df[['Close']].reset_index()
    df_past.rename(columns={'index': 'Date', 'Close': 'Actual'}, inplace=True)
    df_past['Date'] = pd.to_datetime(df_past['Date'])
    df_past['Forecast'] = np.nan
    df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]

    df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast', 'Profit'])
    df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
    print(Y_)
    print(Y_.shape)
    print(len(Y_))
    df_future['Forecast'] = Y_.flatten()
    df_future['Actual'] = np.nan
    df_future['Profit'] = np.nan
    results = pd.concat([df_past, df_future]).set_index('Date')
    #results = df_past.append(df_future).set_index('Date')
    print(results)
    return results
    # # Accuracy
    # fig = plt.figure(figsize=(20, 7))
    # fig.add_subplot(121)
    # plt.plot(history.epoch, history.history['root_mean_squared_error'], label="rmse")
    # plt.title("RMSE", fontsize=18)
    # plt.xlabel("Epochs", fontsize=15)
    # plt.ylabel("RMSE", fontsize=15)
    # plt.grid(alpha=0.3)
    # plt.legend()
    # plt.show()


def get_profit_1(results, input_days):
    sell_date_time = datetime.today() + timedelta(days=input_days)
    sell_date = sell_date_time.strftime('%Y-%m-%d')
    print(sell_date)
    today_1 = datetime.today().strftime('%Y-%m-%d')
    print(today_1)
    const = results['Actual'].loc[today_1]
    results['Profit'] = results['Forecast'] - const
    results['Profit_%'] = results['Profit'] / const * 100
    sell_profit = results['Profit_%'].loc[sell_date]
    print(sell_profit)
    print(results)
    return sell_profit




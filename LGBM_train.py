# connect this to Tkinter, try other models
# modify the def function, order or sth
import pandas as pd
from data_get import preprocess_1
import datetime
import numpy as np
import lightgbm as ltb
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from datetime import datetime, timedelta


# yfin.pdr_override()
# start_day = datetime.datetime(2022, 1, 1)
# end_day = datetime.date.today() + datetime.timedelta(days=1)
#
# #df = pdr.get_data_yahoo(['BTC-USD', 'LTC-USD'], start=start_day, end=end_day)
# df = yfin.download('BTC-USD LTC-USD',
#                    start=start_day, end=end_day, group_by='tickers') # in df, collect all coins (10-50 by market cap)
# #print(df.info()) # date is already df's index
# idx = pd.IndexSlice
# df = df.loc[:,idx['BTC-USD', :]]
# #B = df.loc[:,idx[:,'B']]
# df.columns = ['Open', 'High','Low','Close','Adj Close','Volume']
# df.drop('Adj Close', axis=1, inplace=True)
# #print(df)
# # now the df is ready => we have Open, high, low, close, volume for a selected coin.

# def preprocess(df):
#     #create hour, day and month variables from datetime index
#     # df['hour'] = df.index.hour # not needed
#     df['day'] = df.index.day
#     df['month'] = df.index.month
#     # df['weekday'] = df.index.weekday # this feature was eliminated due to low importance (backward f. elimination)
#
#     #create 1 week lag variable by shifting the target value for 1 week
#     #df['prev_day'] = df['Close'].shift(1)
#     df['lag_1w_Close'] = df['Close'].shift(7)
#     df['lag_2w_Close'] = df['Close'].shift(14)
#     df['lag_1m_Close'] = df['Close'].shift(30)
#     return df



#preprocess(df)

#print(df)
#select interval (day or week or month)
# interval_1 = input('Select the prediction interval (day, week or month): ')
# also select the profit they want to get - another function to suggest the coins that will get this profit
# also 'What-if' questions - what will be the profit if 0.01 BTC is sold for $10000.


def lgbm_train(df, horizon=7):
    df = df.dropna(axis=0)
    X = df.drop(['Close', 'Volume', 'Low', 'High', 'Open'], axis=1)
    y = df['Close']
    # take last week of the dataset for validation
    X_train, X_test = X.iloc[:-horizon, :], X.iloc[-horizon:, :]
    y_train, y_test = y.iloc[:-horizon], y.iloc[-horizon:]
    # create, train and do inference of the model
    model = ltb.LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    # final_model = 'final_model.sav' # ne nado, t.k. when predicting future we need to retrain on all data
    # joblib.dump(model, final_model)
    # calculate MAE
    mape = np.round(mean_absolute_percentage_error(y_test, predictions), 3)
    # plot reality vs prediction for the last week of the dataset
    # fig = plt.figure(figsize=(16, 8))
    # plt.title(f'Real vs Prediction - MAE {mae}', fontsize=20)
    # plt.plot(y_test, color='red')
    pred1 = pd.Series(predictions, index=y_test.index)
    # plt.plot(pred1, color='green')
    # plt.xlabel('Date', fontsize=16)
    # plt.ylabel('Close price', fontsize=16)
    # plt.legend(labels=['Real', 'Prediction'], fontsize=16)
    # plt.grid()
    # plt.show()

    # create a dataframe with the variable importances of the model
    df_importances = pd.DataFrame({
        'feature': model.feature_name_,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    return y_test, pred1, mape, df_importances
    # plot variable importances of the model
    # plt.title('Variable Importances', fontsize=16)
    # sns.barplot(x=df_importances.importance, y=df_importances.feature, orient='h')
    # plt.show()


def lgbm_train_forecast(df, horizon=7, input_date=7):
    X_all = df.drop(['Close','Open','High','Low','Volume'], axis=1)
    y_all = df['Close']

    # train on all data
    model = ltb.LGBMRegressor(random_state=1)
    model.fit(X_all, y_all) #eval_set=[(X_all, y_all)]
    #predictions = model.predict(X_test)
    print(df.index.max())
    print(df.index.max()+pd.Timedelta(days=1))
    # Create future dataframe
    future = pd.date_range(df.index.max()+pd.Timedelta(days=1), df.index.max()+pd.Timedelta(days=input_date), freq='1d')
    future_df = pd.DataFrame(index=future)
    future_df['isFuture'] = True
    df['isFuture'] = False
    df_and_future = pd.concat([df, future_df])
    df_and_future = preprocess_1(df_and_future)
    print(df_and_future)
    future_w_features = df_and_future.query('isFuture').copy()
    a = future_w_features.drop(['Close','Open','High','Low','Volume','isFuture'], axis=1)
    a['pred'] = model.predict(a)
    print(a)
    print(type(a.index))
    return a


def get_profit_2(results, input_days): # need to customise this to show profit like in LSTM. Get acrual today's price for coin.
    sell_date_time = datetime.today() + timedelta(days=input_days)
    sell_date = sell_date_time.strftime('%Y-%m-%d')
    print(sell_date)
    today_1 = datetime.today().strftime('%Y-%m-%d')
    print(today_1)
    #Close
    #const = results['Actual'].loc[today_1]
    results['Profit'] = results['pred'] - results['Actual']
    results['Profit_%'] = results['Profit'] / results['Actual'] * 100 # results['Actual'] is a const (today's price)
    sell_profit = results['Profit_%'].loc[sell_date]
    print(sell_profit)
    print(results)
    return sell_profit


# df = load_data_2()  # there are some coins with NaNs in old dates
# print(df)


#train_time_series_with_folds_2(df)

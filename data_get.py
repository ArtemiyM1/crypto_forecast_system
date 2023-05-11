# pdr can collect only daily data, not hourly
# connect this to Tkinter, try other models
# modify the def function, order or sth
import pandas as pd
import datetime
from yahooquery import Screener
import yfinance as yfin

# yfin.pdr_override()
# start_day = datetime.datetime(2022, 1, 1)
# end_day = datetime.date.today() + datetime.timedelta(days=1)
# #df = pdr.get_data_yahoo(['BTC-USD', 'LTC-USD'], start=start_day, end=end_day)
# s = Screener()
# data = s.get_screeners('all_cryptocurrencies_us', count=50)
# # data is in the quotes key
# dicts = data['all_cryptocurrencies_us']['quotes']
# symbols = [d['symbol'] for d in dicts]
# print(symbols)
# print(len(symbols))
#
# df = yfin.download(symbols, start=start_day, end=end_day, group_by='tickers')
#
# # print(type(df)) # DataFrame
# # print(df.info()) # date is already df's index
# # print(df.columns)
# idx = pd.IndexSlice
#
# A = df.loc[:,idx[:, 'Close']]
# #B = df.loc[:,idx[:,'B']]
# A.columns = A.columns.droplevel(1) # drop Close level => it's a normal DF now
# print(A)
# print(A.info())
# print(A.columns)
# #returns_df = A.pct_change() # there is difference between absolute and % values correlation
# corr = A.dropna().corr()
# print(corr) # we have enough negative correlations
# print(type(corr['BTC-USD'])) # pd series

# selected_corr = corr['BTC-USD'].sort_values(ascending=False)
# print(selected_corr) # we have a sorted series with correlations.
# # choose top 10 / -10 from series:
# print(selected_corr[0:5])
# print(selected_corr[-5:])
# # get names from here - row names/indexes
# print(selected_corr[-5:].index.tolist()) # list of negatively correlated coins. If there are no negative correlations,
# # then it will show the 5 lowest correlations
# print(selected_corr[-5:].values)


# def load_data_1():
#     yfin.pdr_override()
#     start_day = datetime.datetime(2022, 1, 1)
#     end_day = datetime.date.today() + datetime.timedelta(days=1)
#     s = Screener()
#     data = s.get_screeners('all_cryptocurrencies_us', count=50)
#     # data is in the quotes key
#     dicts = data['all_cryptocurrencies_us']['quotes']
#     symbols = [d['symbol'] for d in dicts]
#     # print(symbols)
#     df = yfin.download(symbols, start=start_day, end=end_day, group_by='tickers')
#     idx = pd.IndexSlice
#     df = df.loc[:, idx[:, 'Close']]
#     # B = df.loc[:,idx[:,'B']]
#     df.columns = df.columns.droplevel(1)  # drop Close level => it's a normal DF now
#     return df


def load_data_1():
    yfin.pdr_override()
    start_day = datetime.datetime(2022, 1, 1)
    end_day = datetime.date.today() + datetime.timedelta(days=1)
    s = Screener()
    data = s.get_screeners('all_cryptocurrencies_us', count=50)
    dicts = data['all_cryptocurrencies_us']['quotes']
    symbols = [d['symbol'] for d in dicts]
    df = yfin.download(symbols, start=start_day, end=end_day, group_by='tickers')
    return df, symbols


def preprocess_1(df):
    #create day, week and month variables from datetime index
    df['day'] = df.index.day
    #df['week'] = df.index.week
    df['month'] = df.index.month
    # df['weekday'] = df.index.weekday # this feature was eliminated due to low importance (backward f. elimination)
    df['week'] = pd.Int64Index(df.index.isocalendar().week)
    #create 1 week lag variable by shifting the target value for 1 week
    #df['prev_day'] = df['Close'].shift(1)
    df['lag1w_Clo'] = df['Close'].shift(7)
    df['lag2w_Clo'] = df['Close'].shift(14)
    df['lag1m_Clo'] = df['Close'].shift(30)

    df['lag1w_Hig'] = df['High'].shift(7)
    df['lag2w_Hig'] = df['High'].shift(14)
    df['lag1m_Hig'] = df['High'].shift(30)

    df['lag1w_Low'] = df['Low'].shift(7)
    df['lag2w_Low'] = df['Low'].shift(14)
    df['lag1m_Low'] = df['Low'].shift(30)

    df['lag1w_Vol'] = df['Volume'].shift(7)
    df['lag2w_Vol'] = df['Volume'].shift(14)
    df['lag1m_Vol'] = df['Volume'].shift(30)
    return df


import numpy as np
X=[]
Y=[]
list1=[1,2,3,4,5,6,7,8,9,10]
list1 = list(np.arange(1, 400))
#print(list1)
for i in range(60, len(list1) - 30 + 1):
    X.append(list1[i - 60: i])
    Y.append(list1[i: i + 30])

print(X)
print(Y)
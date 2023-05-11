# https://thedefiant.io/feed/
# https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml
# https://cointelegraph.com/rss
# https://cryptopotato.com/feed/
# https://cryptoslate.com/feed/
# https://cryptonews.com/news/feed/
# https://smartliquidity.info/feed/
# https://finance.yahoo.com/news/rssindex
# https://www.cnbc.com/id/10000664/device/rss/rss.html
# https://time.com/nextadvisor/feed/
# https://benjaminion.xyz/newineth2/rss_feed.xml


# import tkinter
# import  feedparser
# import numpy as np
#
# window = tkinter.Tk()
#
# """Edit Window"""
# window.title("Notebook") #https://hackernoon.com/tagged/hackernoon-top-story/feed
# window.geometry("1000x1000")
#
# rssfeed = tkinter.Frame(window, bg='black', width=200, height=80)
# feed = feedparser.parse('https://coinpedia.org/feed/')
# feedShow = {'entries': [{feed['entries'][0]['title']}]}
# print(feedShow)
#
# lis = []
# print(np.arange(1, 6))
# for i in range(1, 10):
#     a = feed['entries'][i]['title']
#     lis.append(a)
# print(lis)
#
#
# class RSSDisplay(tkinter.Frame):
#     def __init__(self, master=None, **kw):
#         tkinter.Frame.__init__(self, master=master, **kw)
#         self.txtHeadline = tkinter.StringVar()
#         self.headline = tkinter.Label(self, textvariable=self.txtHeadline,
#                                       bg='black', fg='white', font=("arial", 20))
#         self.headline.grid()
#         self.headlineIndex = 0
#         self.updateHeadline()
#
#     def updateHeadline(self):
#         try:
#             headline = feed['entries'][self.headlineIndex]['title']
#         except IndexError:
#             self.headlineIndex = 0
#             headline = feed['entries'][self.headlineIndex]['title']
#         self.txtHeadline.set(headline)
#         self.headlineIndex += 1
#         self.after(10000, self.updateHeadline)
#
#
# RSSDisplay(window).pack(expand='yes', fill='x')
#
# tkinter.mainloop()







# df = pd.read_csv("results_2", index_col=False)
# print(df)

# for ind in df.index:
#     print(df['Title'][ind], df['Link'][ind])
#     link = '{0} {1}'.format("Link:", df['Link'][ind])
#     title = '{0} {1}'.format("\nTitle:", df['Title'][ind])
# for row in df.index:
#     link = '{0} {1}'.format("Link:", row['Link'])
#     title = '{0} {1}'.format("\nTitle:", row['Title'])
#     # votes = '{0} {1}'.format("\nVotes:", row['votes'])
#     # text.insert(INSERT, link)
#     # text.insert(INSERT, title)
#     # text.insert(INSERT, votes)
#     # text.insert(END, "\n\n")
# print(link)
# print(title)


#
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 30)
# pd.set_option('display.width', 10000)
# pd.set_option('display.max_colwidth', None)
#
# results = ['results_feed_blog.buyucoin.com.csv', 'results_feed_coinpedia.org.csv', 'results_feed_cryptopotato.com.csv', 'results_feed_u.today.csv', 'results_feed_www.coindesk.com.csv']
# df_from_each_file = (pd.read_csv(f) for f in results)
# concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
# concatenated_df = concatenated_df[['Title', 'Link']]
# concatenated_df = concatenated_df.reset_index(drop=True)
# print(concatenated_df)
# concatenated_df.to_csv("results_2", index=False)



# import pandas
# import tkinter as tk
# #creates the dataframe
#
# window = tk.Tk() #start of the main window
# #function that will search the dataframe column "company" for any matches
# # df = pd.read_csv("results_feed_1")
# # print(df)
#
# def search_df(*event):
#     search_result=df.loc[df['Title'].str.contains(e1_value.get(),
#                                na=False, #ignore the cell's value is Nan
#                                case=False)] #case insensitive
#     t1.insert(tk.END, search_result)
#
#
# #Creates the entry box and link the e1_value to the variable
# e1_value=tk.StringVar()
# e1=tk.Entry(window, textvariable=e1_value)
# e1.grid(row=0,column=0)
# #execute the search_df function when you hit the "enter" key and put an event
# #parameter
# e1.bind("<Return>", search_df)
#
# #Creates a button
# b1=tk.Button(window,
#              width=10,
#              text='search',
#              command=search_df)
#
# b1.grid(row=0,column=1)
#
# #Creates a text box
# t1=tk.Text(window,height=5,width=80)
# t1.grid(row=0,column=2)
#
# window.mainloop() #end of the main window





# #Import tkinter library
# from tkinter import *
# from tkinter import ttk
# #Create an instance of Tkinter frame or window
# win= Tk()
# #Set the geometry of tkinter frame
# win.geometry("750x250")
# #Create a text widget and wrap by words
# text= Text(win,wrap=WORD)
# text.insert(INSERT,"Python is an interpreted, high-level and general-purpose programming language. Python's design philosophy emphasizes code readability with its notable use of significant indentation.")
# text.pack()
# win.mainloop()

# import numpy as np
# list1 = list(np.arange(1, 31))
# a=[]
# for i in list1:
#     output = '{0}'.format(list1[i-1])
#     a.append(output)
# print(a)




# import requests
# from bs4 import BeautifulSoup
# import pandas as pd
#
#
#
# def get_df_RSS():
#     feeds = ['https://smartliquidity.info/feed/', 'https://finance.yahoo.com/news/rssindex',
#              'https://blog.buyucoin.com/feed/', 'https://cointelegraph.com/rss/tag/altcoin',
#              'https://cryptopotato.com/feed/', 'https://cointelegraph.com/rss/category/top-10-cryptocurrencies',
#              'https://cointelegraph.com/rss/tag/regulation', 'https://cointelegraph.com/rss',
#              'https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml',
#              'https://u.today/rss']  # 'https://coinpedia.org/feed/'
#     output = []
#
#     for url in feeds:
#         resp = requests.get(url)
#         soup = BeautifulSoup(resp.text, 'xml')
#
#         for entry in soup.find_all('item'):
#             item = {
#                 'Title': entry.find('title').text,
#                 # 'Pubdate': e.text if(e := entry.find('pubDate')) else None,
#                 # 'Content': entry.find('description').text,
#                 'Link': entry.find('link').text
#             }
#
#             output.append(item)
#
#     df = pd.DataFrame(output)
#     print(df)
#     return df


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

# pd.options.mode.chained_assignment = None
# tf.random.set_seed(0)
# from numpy.random import seed
# seed(1)
# tf.random.set_seed(221)
#
# from data_get import load_data_1  # do like this
# from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
# from sklearn.preprocessing import MinMaxScaler
#
# df = load_data_1()  # there are some coins with NaNs in old dates
# # # plotting the graph
# idx = pd.IndexSlice
# df = df.loc[:, idx['BTC-USD', :]]  # coin
# df.columns = df.columns.droplevel(0)
# df = df.drop('Adj Close', axis=1)
# #df1 = preprocess_1(df1)
# print(df)
# print(df['Close'].values.shape)


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

# train_forecast1(df)



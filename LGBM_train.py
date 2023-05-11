import pandas as pd
from data_get import preprocess_1
import datetime
import numpy as np
import lightgbm as ltb
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from datetime import datetime, timedelta


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
    # calculate MAE
    mape = np.round(mean_absolute_percentage_error(y_test, predictions), 3)

    pred1 = pd.Series(predictions, index=y_test.index)

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
    model.fit(X_all, y_all)
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


def get_profit_2(results, input_days): 
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


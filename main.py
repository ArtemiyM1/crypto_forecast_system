import tkinter.ttk as tkrtk
import tkinter as tkr
import customtkinter
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import tkcalendar
import pandas as pd
from data_get import load_data_1, preprocess_1
from LGBM_train import lgbm_train, get_profit_2, lgbm_train_forecast
from LSTM_forecast import train_forecast1, get_profit_1
from RSS_feed import get_df_RSS
from numpy.random import seed
import tensorflow as tf

seed(1)
tf.random.set_seed(221)

df, symbols = load_data_1()
df_rss = get_df_RSS()
print(df)
print(symbols)


def plot_1():
    try:
        global canvas
        canvas.get_tk_widget().pack_forget()
        global toolbar
        toolbar.destroy()
        print("Plot Page has been cleared")
    except NameError:
        pass
    # the figure that will contain the plot
    fig = Figure(figsize=(5, 5), dpi=100)
    # adding the subplot
    plot1 = fig.add_subplot(111)
    fig.autofmt_xdate(rotation=30)
    # plotting the graph
    idx = pd.IndexSlice
    df1 = df.loc[:, idx[coin, :]]
    df1 = df1[input_date1:input_date2]
    df1.columns = df1.columns.droplevel(0)
    print(df1)
    plot1.plot(df1['Close'])
    plot1.grid()
    # plot1.plot(df.index, df['BTC-USD'])
    # creating the Tkinter canvas containing the Matplotlib figure
    plot1.set_title(f'Close price for the chosen coin and period ({coin})')

    canvas = FigureCanvasTkAgg(fig, master=sheet1)
    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().pack()
    canvas.draw()
    # creating the Matplotlib toolbar
    toolbar = NavigationToolbar2Tk(canvas, sheet1)
    toolbar.update()
    # placing the toolbar on the Tkinter window
    canvas.get_tk_widget().pack()
    # toolbar.destroy()
    # button_1['state'] = "disabled"
    # clear(canvas=canvas)
    # canvas.delete("all")
    # canvas.get_tk_widget().pack_forget()

def plot_2():
    try:
        global canvas
        canvas.get_tk_widget().pack_forget()
        global toolbar
        toolbar.destroy()
        print("Plot Page has been cleared")
    except NameError:
        pass
    fig = Figure(figsize=(5, 5), dpi=100)
    plot1 = fig.add_subplot(111)
    fig.autofmt_xdate(rotation=30)
    # plotting the graph
    idx = pd.IndexSlice
    df1 = df.loc[:, idx[coin, :]]
    df1 = df1[input_date1:input_date2]
    df1.columns = df1.columns.droplevel(0)

    df1['MA 3 days'] = df1['Close'].rolling(window=3, center=False).mean()
    df1['MA 14 days'] = df1['Close'].rolling(window=14, center=False).mean()
    t1, = plot1.plot(df1.index, df1['Close'])
    t2, = plot1.plot(df1.index, df1['MA 3 days'])
    t3, = plot1.plot(df1.index, df1['MA 14 days'])
    print(df1)
    plot1.grid()
    fig.legend((t1,t2,t3), ('Close', 'MA 3 days', 'MA 14 days'), 'upper right')
    plot1.set_title(f'MA for the chosen coin ({coin})')
    canvas = FigureCanvasTkAgg(fig, master=sheet2)
    canvas.get_tk_widget().pack()
    canvas.draw()
    toolbar = NavigationToolbar2Tk(canvas, sheet2)
    toolbar.update()
    canvas.get_tk_widget().pack()


def plot_3(): #3rd sheet correlations
    try:
        global canvas
        canvas.get_tk_widget().pack_forget()
        global toolbar
        toolbar.destroy()
        print("Plot Page has been cleared")
    except NameError:
        pass
    fig = Figure(figsize=(8, 5), dpi=100) # figsize=5,5
    plot1 = fig.add_subplot(111)
    #fig.autofmt_xdate(rotation=45)
    # plotting the graph
    idx = pd.IndexSlice
    global df
    df1 = df.copy()
    df1 = df1[input_date1:input_date2]
    df1 = df1.loc[:, idx[:, 'Close']]
    df1.columns = df1.columns.droplevel(1)
    print(df1)
    corr = df1.corr()
    selected_corr = corr[coin].sort_values(ascending=False).dropna(axis=0)
    print(selected_corr)
    a = selected_corr[0:10]
    b = selected_corr[-10:]
    df2 = pd.concat([a, b], axis=1, ignore_index=True, sort=False)
    df2.columns = ['Top 10 positively correlated', 'Top 10 negatively correlated\n (or least correlated)']
    print(df2)
    df2.plot(kind='barh', ax=plot1, title=f'Most correlated coins (with {coin}) for the chosen period').set_xlabel("Correlation coefficient")

    canvas = FigureCanvasTkAgg(fig, master=sheet3)
    canvas.get_tk_widget().pack(side="top")# fill='both', expand=True)
    canvas.draw()
    toolbar = NavigationToolbar2Tk(canvas, sheet3)
    toolbar.update()
    canvas.get_tk_widget().pack()


def plot_4():
    try:
        global canvas
        canvas.get_tk_widget().pack_forget()
        global toolbar
        toolbar.destroy()
        print("Plot Page has been cleared")
    except NameError:
        pass
    fig = Figure(figsize=(5, 5), dpi=100)
    plot1 = fig.add_subplot(111)
    fig.autofmt_xdate(rotation=30)
    # plotting the graph
    idx = pd.IndexSlice
    df1 = df.loc[:, idx[coin, :]]
    #df1 = df1[input_date1:input_date2]
    df1.columns = df1.columns.droplevel(0)
    df1 = df1.drop('Adj Close', axis=1)
    preprocess_1(df1)
    y_test, pred1, mape, df_importances = lgbm_train(df1, horizon=7)# choose big interval (input dates) for the model to train normally. Enough training data
    print(y_test, pred1)
    t1, = plot1.plot(y_test, color='red')
    t2, = plot1.plot(pred1, color='green')
    fig.legend((t1,t2), ('Real','Prediction'), 'upper left')
    plot1.set_title(f'Real vs Prediction - MAPE {mape}')
    plot1.grid()

    # from scipy.signal import find_peaks
    # # x = np.array([6, 3, 5, 2, 1, 4, 9, 7, 8])
    # # y = np.array([2, 1, 3, 5, 3, 9, 8, 10, 7])
    # peaks, _ = find_peaks(y)
    # # this way the x-axis corresponds to the index of x
    # plt.plot(x - 1, y)
    # plt.plot(peaks, y[peaks], "x")
    # plt.show()

    canvas = FigureCanvasTkAgg(fig, master=sheet4)
    canvas.get_tk_widget().pack()
    canvas.draw()
    toolbar = NavigationToolbar2Tk(canvas, sheet4)
    toolbar.update()
    canvas.get_tk_widget().pack()


def plot_5():
    try:
        global canvas
        canvas.get_tk_widget().pack_forget()
        global toolbar
        toolbar.destroy()
        print("Plot Page has been cleared")
    except NameError:
        pass
    fig = Figure(figsize=(7, 5), dpi=100)
    plot1 = fig.add_subplot(111)
    fig.autofmt_xdate(rotation=30)
    # plotting the graph
    idx = pd.IndexSlice
    df1 = df.loc[:, idx[coin, :]]
    df1.columns = df1.columns.droplevel(0)
    df1 = df1.drop('Adj Close', axis=1)
    preprocess_1(df1)
    y_test, pred1, mae, df_importances = lgbm_train(df1, horizon=7)
    df_importances.set_index("feature", inplace=True)
    print(df_importances)
    df_importances.plot(kind='barh', legend=False, ax=plot1)
    plot1.set_title('Feature importance by LightGBM')

    canvas = FigureCanvasTkAgg(fig, master=sheet5)
    canvas.get_tk_widget().pack()
    canvas.draw()
    toolbar = NavigationToolbar2Tk(canvas, sheet5)
    toolbar.update()
    canvas.get_tk_widget().pack()


def plot_6():
    try:
        global canvas
        canvas.get_tk_widget().pack_forget()
        global toolbar
        toolbar.destroy()
        print("Plot Page has been cleared")
    except NameError:
        pass
    fig = Figure(figsize=(5, 5), dpi=100)
    plot1 = fig.add_subplot(111)
    fig.autofmt_xdate(rotation=30)
    # plotting the graph
    idx = pd.IndexSlice
    global df1
    df1 = df.loc[:, idx[coin, :]] # coin
    df1.columns = df1.columns.droplevel(0)
    df1 = df1.drop('Adj Close', axis=1)
    df1 = preprocess_1(df1)
    global df2
    df2 = lgbm_train_forecast(df1, input_date=input_date3)
    #df1 = train_forecast1(df1)

    # results.plot(title='BTC')
    plot1.plot(df2.index, df2['pred'])
    plot1.grid()
    plot1.set_title(f'{input_date3}-day Forecast by LightGBM')

    canvas = FigureCanvasTkAgg(fig, master=sheet6)
    canvas.get_tk_widget().pack()
    canvas.draw()
    toolbar = NavigationToolbar2Tk(canvas, sheet6)
    toolbar.update()
    canvas.get_tk_widget().pack()
    return df1, df2


def pack_profit_2(): #we have two dfs accessible now. Need to concate4nate them to get today's coin price.
    from datetime import datetime, timedelta
    global df1
    today_1 = datetime.today().strftime('%Y-%m-%d')
    print(today_1)
    const = df1['Close'].loc[today_1]
    global df2
    df2['Actual'] = const
    print(df2)
    pp = get_profit_2(df2, input_date3)
    print(pp)
    tab6_label = tkr.Label(sheet6, text=f'Your profit/loss per 1 coin if you buy {coin} today and \n sell it in {input_date3} days: {pp:.2f} %')
    tab6_label.place(in_=button_9, relx=1.0, x=20, rely=0)


def plot_7():
    try:
        global canvas
        canvas.get_tk_widget().pack_forget()
        global toolbar
        toolbar.destroy()
        print("Plot Page has been cleared")
    except NameError:
        pass
    fig = Figure(figsize=(5, 5), dpi=100)
    plot1 = fig.add_subplot(111)
    fig.autofmt_xdate(rotation=30)
    # plotting the graph
    idx = pd.IndexSlice
    global df1
    df1 = df.loc[:, idx[coin, :]] # coin
    df1.columns = df1.columns.droplevel(0)
    df1 = df1.drop('Adj Close', axis=1)
    df1 = preprocess_1(df1)

    #df1 = train_time_series_with_folds_2(df1)
    df1 = train_forecast1(df1)
    #input_days = 7
    # global p
    # p = get_profit(df1, input_date3)
    # print(p)
    plot1.plot(df1.index, df1['Forecast'])
    plot1.grid()
    plot1.set_title(f'30-day forecast by LSTM ({coin})')

    canvas = FigureCanvasTkAgg(fig, master=sheet7)
    canvas.get_tk_widget().pack()
    canvas.draw()
    toolbar = NavigationToolbar2Tk(canvas, sheet7)
    toolbar.update()
    canvas.get_tk_widget().pack()
    return df1 #, p # was without p


def pack_profit():
    global df1
    print(df1)
    p = get_profit_1(df1, input_date3)
    print(p)
    tab7_label = tkr.Label(sheet7, text=f'Your profit/loss if you buy 1 {coin} today and \n sell it in {input_date3} days: {p:.2f} %')
    tab7_label.place(in_=button_11, relx=1.0, x=20, rely=0) # in_ operator is to place widget relative to a specified widget


def search_df(*event):
    try:
        t1.delete(1.0, "end")
        print("Text box has been cleared")
    except NameError:
        pass
    pd.set_option('display.max_colwidth', 100)
    #df = pd.read_csv("results_2", index_col=False)
    # df_rss = get_df_RSS()
    # print(df_rss)
    search_result=df_rss.loc[df_rss['Title'].str.contains(e1_value.get(),
                               na=False, #ignore the cell's value is Nan
                               case=False)] #case insensitive
    for ind in search_result.index:
        #print(search_result['Title'][ind], search_result['Link'][ind])
        title = '{0} {1}'.format("Title:", search_result['Title'][ind])
        link = '{0} {1}'.format("\nLink:", search_result['Link'][ind])
        t1.insert(tkr.INSERT, title)
        t1.insert(tkr.INSERT, link)
        t1.insert(tkr.END, "\n\n")


def callback(selection):
    global coin  # we have global coin, input_date1 and input_date2 ready to use in all sheets
    coin = selection
    print(coin)
    return coin


def callback_date1():
    global input_date1
    input_date1 = date1.get_date()
    print(input_date1)
    return input_date1


def callback_date2():
    global input_date2
    input_date2 = date2.get_date()
    print(input_date2)
    return input_date2


def callback_date3(selection):
    global input_date3  # we have global coin, input_date1 and input_date2 ready to use in all sheets
    input_date3 = int(selection)
    print(input_date3)
    return input_date3


# open, close, high, low are highly correlated so we dont use them all. only close
# light GBM is faster than LSTM so we use it. cause we need to retrain model in real time
# do scaling and random search for hyperparameter tuning, maybe with RF do ensemble

"""Create Window"""
window = tkr.Tk()

"""Edit Window"""
window.title("Notebook")
window.geometry("1000x1000")

"""Create Notebook"""
notebook = tkrtk.Notebook(window)

def my_msg(*args):
    tab_nos = str(notebook.index(notebook.select())+1)
    l4.config(text='sheet No: ' + tab_nos)

notebook.bind('<<NotebookTabChanged>>', my_msg)
l4=tkr.Label(window,text='')
l4.pack(side=tkr.LEFT)

"""Create tabs"""
sheet1 = tkrtk.Frame(notebook)
sheet2 = tkrtk.Frame(notebook)
sheet3 = tkrtk.Frame(notebook)
sheet4 = tkrtk.Frame(notebook)
sheet5 = tkrtk.Frame(notebook)
sheet6 = tkrtk.Frame(notebook)
sheet7 = tkrtk.Frame(notebook)
sheet8 = tkrtk.Frame(notebook)

sheet1.pack(fill='both', expand=1)
sheet2.pack(fill='both', expand=1)
sheet3.pack(fill='both', expand=1)
sheet4.pack(fill='both', expand=1)
sheet5.pack(fill='both', expand=1)
sheet6.pack(fill='both', expand=1)
sheet7.pack(fill='both', expand=1)
sheet8.pack(fill='both', expand=1)

"""Add tabs to Notebook"""
notebook.add(sheet1, text="sheet1")
notebook.add(sheet2, text="sheet2")
notebook.add(sheet3, text="sheet3")
notebook.add(sheet4, text="sheet4")
notebook.add(sheet5, text="sheet5")
notebook.add(sheet6, text="sheet6")
notebook.add(sheet7, text="sheet7")
notebook.add(sheet8, text="sheet8")
notebook.pack(fill="both", expand="yes")

"""Add Widgets to each tab"""
tab1_label = tkr.Label(sheet1, text="Welcome to Sheet1 \n This program helps analyse the crypto market. \n Press Ctrl+Tab to switch between sheets.")
tab1_label.pack()
label1 = tkr.Label(sheet1, text='Choose cryptocurrency symbol from the list: ')
label1.pack()
# choose coin
optionmenu_1 = customtkinter.CTkOptionMenu(sheet1, values=symbols, command=callback)
optionmenu_1.pack(pady=10, padx=10)
optionmenu_1.set("Select")
label2 = tkr.Label(sheet1, text='Choose start date: ')
label2.pack()
date1 = tkcalendar.DateEntry(sheet1, command=callback_date1)
date1.pack(padx=10, pady=10)
button_1 = tkr.Button(sheet1, text="Confirm start date", command=callback_date1)
button_1.pack()
label3 = tkr.Label(sheet1, text='Choose end date: ')
label3.pack()
date2 = tkcalendar.DateEntry(sheet1, command=callback_date2)
date2.pack(padx=10, pady=10)
button_2 = tkr.Button(sheet1, text="Confirm end date", command=callback_date2)
button_2.pack()
button_3 = tkr.Button(sheet1, text="Show Close price", command=plot_1)
button_3.pack(pady=10, padx=10)

tab2_label = tkr.Label(sheet2, text="Welcome to Sheet2")
tab2_label.pack()
tab2_label = tkr.Label(sheet2, text="Press button to plot Moving Average")
tab2_label.pack()
button_4 = tkr.Button(sheet2, text="Plot MA", command=plot_2)
button_4.pack()

tab3_label = tkr.Label(sheet3, text="Welcome to Sheet3")
tab3_label.pack()
tab3_label = tkr.Label(sheet3, text="Press button to display correlated coins")
tab3_label.pack()
button_5 = tkr.Button(sheet3, text="Correlations", command=plot_3)
button_5.pack()

tab4_label = tkr.Label(sheet4, text="Welcome to Sheet4 \n Press button to view model fitting and accuracy")
tab4_label.pack()
button_6 = tkr.Button(sheet4, text="Model fitting", command=plot_4)
button_6.pack()

tab5_label = tkr.Label(sheet5, text="Welcome to Sheet5 \n Press button to view the model's feature importance")
tab5_label.pack()
button_7 = tkr.Button(sheet5, text="Feature importance", command=plot_5)
button_7.pack()


tab6_label = tkr.Label(sheet6, text="Welcome to Sheet6 \n Press button to predict the price \n for the selected number of days. \n\n Select number of days to forecast: ")
tab6_label.pack()
# date3 = tkr.Entry(sheet6) # unknown option "-command" for entries (also DateEntries)
# date3.pack(padx=10, pady=10)
list1 = list(np.arange(2, 31))
coin_symbols=[]
for i in list1:
    output = '{0}'.format(list1[i-2])
    coin_symbols.append(output)
print(coin_symbols)
optionmenu_2 = customtkinter.CTkOptionMenu(sheet6, values=coin_symbols[0:6], command=callback_date3)
optionmenu_2.pack(pady=10, padx=10)
optionmenu_2.set("Select days")
button_8 = tkr.Button(sheet6, text="Predict future price", command=plot_6)
button_8.pack()
tab6_label = tkr.Label(sheet6, text=f"Select in how many days (from now) you want to sell your coin: \n (this number must be less/equal to the number of days to forecast)")
tab6_label.pack()
optionmenu_3 = customtkinter.CTkOptionMenu(sheet6, values=coin_symbols[0:6], command=callback_date3)
optionmenu_3.pack(pady=10, padx=10)
optionmenu_3.set("Select days")
button_9 = tkr.Button(sheet6, text="Calculate profit", command=pack_profit_2)
button_9.pack()



tab7_label = tkr.Label(sheet7, text=f"Welcome to Sheet7 \n Press button to view LSTM's 30 day forecast \n (takes around 40 seconds to generate)")
tab7_label.pack()
button_10 = tkr.Button(sheet7, text="LSTM", command=plot_7)
button_10.pack()
label7 = tkr.Label(sheet7, text=f"Select in how many days (from now) you want to sell your coin: ")
label7.pack()
optionmenu_4 = customtkinter.CTkOptionMenu(sheet7, values=coin_symbols, command=callback_date3)
optionmenu_4.pack(pady=10, padx=10)
optionmenu_4.set("Select days")
button_11 = tkr.Button(sheet7, text="Calculate profit", command=pack_profit)
button_11.pack()


tab8_label = tkr.Label(sheet8, text="Welcome to Sheet8 \n Use the search bar to type and get matching results \n For example, type 'BTC' or 'bitcoin' and press Enter")
tab8_label.pack()
#Creates the entry box and link the e1_value to the variable
e1_value=tkr.StringVar()
e1=tkr.Entry(sheet8, textvariable=e1_value)
e1.pack()
#execute the search_df function when you hit the "enter" key and put an event parameter
e1.bind("<Return>", search_df)
button_12 = tkr.Button(sheet8, text="search", command=search_df)
button_12.pack()
#Creates a text box
t1=tkr.Text(sheet8, wrap='word') #height=30,width=1000
t1.pack(fill='both', expand=1)

notebook.enable_traversal() # allows to use ctrl+Tab to switch sheets

"""Activate"""
tkr.mainloop()


# results = ['results_feed_blog.buyucoin.com.csv', 'results_feed_coinpedia.org.csv', 'results_feed_cryptopotato.com.csv', 'results_feed_u.today.csv', 'results_feed_www.coindesk.com.csv']
# df_from_each_file = (pd.read_csv(f) for f in results)
# concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
# concatenated_df = concatenated_df[['Title', 'Link']]
# print(concatenated_df)
# concatenated_df.to_csv("results_2")


#
# import customtkinter
#
# customtkinter.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
# customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"
#
# app = customtkinter.CTk()
# app.geometry("400x780")
# app.title("CustomTkinter simple_example.py")
#
# def button_callback():
#     print("Button click", combobox_1.get())
#
#
# def slider_callback(value):
#     progressbar_1.set(value)
#
#
# frame_1 = customtkinter.CTkFrame(master=app)
# frame_1.pack(pady=20, padx=60, fill="both", expand=True)
#
# label_1 = customtkinter.CTkLabel(master=frame_1, justify=customtkinter.LEFT)
# label_1.pack(pady=10, padx=10)
#
# progressbar_1 = customtkinter.CTkProgressBar(master=frame_1)
# progressbar_1.pack(pady=10, padx=10)
#
# button_1 = customtkinter.CTkButton(master=frame_1, command=button_callback)
# button_1.pack(pady=10, padx=10)
#
# slider_1 = customtkinter.CTkSlider(master=frame_1, command=slider_callback, from_=0, to=1)
# slider_1.pack(pady=10, padx=10)
# slider_1.set(0.5)
#
# entry_1 = customtkinter.CTkEntry(master=frame_1, placeholder_text="CTkEntry")
# entry_1.pack(pady=10, padx=10)
#
# optionmenu_1 = customtkinter.CTkOptionMenu(frame_1, values=["Option 1", "Option 2", "Option 42 long long long..."])
# optionmenu_1.pack(pady=10, padx=10)
# optionmenu_1.set("CTkOptionMenu")
#
# combobox_1 = customtkinter.CTkComboBox(frame_1, values=["Option 1", "Option 2", "Option 42 long long long..."])
# combobox_1.pack(pady=10, padx=10)
# combobox_1.set("CTkComboBox")
#
# checkbox_1 = customtkinter.CTkCheckBox(master=frame_1)
# checkbox_1.pack(pady=10, padx=10)
#
# radiobutton_var = customtkinter.IntVar(value=1)
#
# radiobutton_1 = customtkinter.CTkRadioButton(master=frame_1, variable=radiobutton_var, value=1)
# radiobutton_1.pack(pady=10, padx=10)
#
# radiobutton_2 = customtkinter.CTkRadioButton(master=frame_1, variable=radiobutton_var, value=2)
# radiobutton_2.pack(pady=10, padx=10)
#
# switch_1 = customtkinter.CTkSwitch(master=frame_1)
# switch_1.pack(pady=10, padx=10)
#
# text_1 = customtkinter.CTkTextbox(master=frame_1, width=200, height=70)
# text_1.pack(pady=10, padx=10)
# text_1.insert("0.0", "CTkTextbox\n\n\n\n")
#
# segmented_button_1 = customtkinter.CTkSegmentedButton(master=frame_1, values=["CTkSegmentedButton", "Value 2"])
# segmented_button_1.pack(pady=10, padx=10)
#
# tabview_1 = customtkinter.CTkTabview(master=frame_1, width=200, height=70)
# tabview_1.pack(pady=10, padx=10)
# tabview_1.add("CTkTabview")
# tabview_1.add("Tab 2")
#
# app.mainloop()

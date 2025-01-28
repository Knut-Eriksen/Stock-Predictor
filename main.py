import pandas as pd
from prophet import Prophet
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

#Get ticker from user
def ticker_input():
    ticker = input('Enter ticker symbol: ').upper()
    return ticker

#Add live time as end date and 10 years before as start date
end_date = datetime.today()
start_date = end_date - timedelta(days = 10*365)

#Test using ticker_input with start and end parameters
data = yf.download(ticker_input(), start_date, end_date)

#Reset index of dataframe
data.reset_index(inplace=True)

#Create dataframe
close_df = pd.DataFrame()

#Edit column names to use prophet
close_df["ds"] = data["Date"]
close_df["y"] = data["Close"]

#Initiate Prophet and start fit
model = Prophet()
model.fit(close_df)

#Create future dataframe until one year from today
future_df = model.make_future_dataframe(periods=365)

#Debug last 5 objects
print(future_df.tail())

# Make prediction
prediction = model.predict(future_df)

#Print ds: date, yhat: predicted value, yhat_lower: lower bound of the uncertainty interval, yhat_upper: upper bound of the uncertainty interval
print(prediction[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

#Plot the dataframe and prediction
graph = model.plot(prediction)
plt.show()
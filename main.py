import pandas as pd
from prophet import Prophet
import yfinance as yf
from datetime import datetime, timedelta

#Add live time as end date and 10 years before as start date
end_date = datetime.today()
start_date = end_date - timedelta(days = 10*365)

#Test using QQQ ticker with start and end parameters
data = yf.download("QQQ", start_date, end_date)

#Reset index of dataframe
data.reset_index(inplace=True)

#Create dataframe
close_df = pd.DataFrame()

#Edit column names to use prophet
close_df["ds"] = data["Date"]
close_df["y"] = data["Close"]

print(close_df)

#Initiate Prophet and start fit
m = Prophet()
m.fit(close_df)
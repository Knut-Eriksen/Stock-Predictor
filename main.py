import pandas as pd
from prophet import Prophet

#Read csv
df = pd.read_csv('sp500_index.csv')

#Check csv content
print(df)
print(df.info())

#Initiate Prophet and start fit
m = Prophet()
m.fit(df)


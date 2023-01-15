import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import json
import statsmodels.api as sm
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

datauntil = 12
forecastperiod = 12

data1 = pd.read_csv("tmp.csv")
data1.drop(columns=['na_item'], inplace=True)
data2 = pd.melt(data1, id_vars=['nace_r2', 'geo\TIME_PERIOD'])
data2.columns = ['Sector', 'Region', 'Date', 'Data']


df = data2[data2['Region'] == 'NL']
df = df[df['Sector'] == sector]
df = df.dropna()
endog = df["Data"].values[:-forecastperiod]

########### ARIMA ###########################

# Construct the model
mod = sm.tsa.SARIMAX(endog, order=(1, 0, 0), trend='ct')
# Estimate the parameters
res = mod.fit()
forecasts_arima = res.forecast(steps=forecastperiod)

#############################################

########### ExponentialSmoothing ###########################

fit1 = SimpleExpSmoothing(endog).fit()
expo_SES = fit1.forecast(forecastperiod)
fit2 = Holt(endog).fit()
expo_Holts = fit2.forecast(forecastperiod)
fit3 = Holt(endog, exponential=True).fit()
expo_Expon = fit3.forecast(forecastperiod)
fit4 = Holt(endog, damped_trend=True).fit(damping_slope=0.5)
expo_AdditiveDamp = fit4.forecast(forecastperiod)
fit5 = Holt(endog, exponential=True, damped_trend=True).fit()
expo_MulitDamp = fit5.forecast(forecastperiod)

#############################################


df['Date'] = pd.PeriodIndex(df.Date, freq='Q').to_timestamp()
df.set_index("Date", inplace=True)
df = pd.pivot(df, columns='Sector', values='Data')


##################################################
##################################################

df['ARIMA_Forecasts'] = np.nan
df.iloc[-forecastperiod:, -1] = forecasts_arima

df['SES'] = np.nan
df.iloc[-forecastperiod:, -1] = expo_SES
df['Holts'] = np.nan
df.iloc[-forecastperiod:, -1] = expo_Holts
df['Expon'] = np.nan
df.iloc[-forecastperiod:, -1] = expo_Expon
df['AdditiveDamp'] = np.nan
df.iloc[-forecastperiod:, -1] = expo_AdditiveDamp
df['MulitDamp'] = np.nan
df.iloc[-forecastperiod:, -1] = expo_MulitDamp

##################################################
##################################################

labels = {
             "Sector": "Natural Log Millions"
         }

fig = px.line(df, x = df.index, y = df.columns, title= "Country: " + country, labels = labels, template="seaborn", width = 1200, height = 800)
fig.update_layout(
    autosize= False,
    margin=dict(l=40, r=20, t=40, b=40),
)
fig.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 1)',
    'paper_bgcolor': 'rgba(128, 128, 128, .3)',
})

fig.update_yaxes(automargin=True)

graphJSON2 = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
return graphJSON2

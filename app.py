from flask import Flask, config, render_template, request
import pandas as pd
import json
import plotly
import plotly.express as px
import numpy as np
from flask_bootstrap import Bootstrap
import statsmodels.api as sm
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

app = Flask(__name__)
bootstrap = Bootstrap(app)

@app.route('/callback1', methods=['POST', 'GET'])
def cb1():
    return gm_sector(request.args.get('data'))

@app.route('/callback2', methods=['POST', 'GET'])
def cb2():
    return gm_country(request.args.get('data'))

@app.route('/callback3', methods=['POST', 'GET'])
def cb3():
    return gm_netherlands(request.args.get('data'))


@app.route("/")
def view_home():
    return render_template("index.html", title="European Sectoral Data")

@app.route('/sector')
def sector():
    return render_template('sector.html', graphJSON=gm_sector(), title="Sectors")

@app.route('/country')
def country():
    return render_template('region.html', graphJSON=gm_country(), title="Regions")

@app.route('/netherlands')
def netherlands():
    return render_template('netherlands.html', graphJSON=gm_netherlands(), title="Netherlands (Sectors)")

def gm_netherlands(sector='A'):

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
        "Sector": "Millions"
    }

    fig = px.line(df, x=df.index, y=df.columns, title="Country: " + "Netherlands", labels=labels, template="seaborn", width=1200, height=800)
    fig.update_layout(
        autosize=False,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 1)',
        'paper_bgcolor': 'rgba(128, 128, 128, .3)',
    })

    fig.update_yaxes(automargin=True)

    graphJSON3 = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON3


def gm_country(country='NL'):

    data1 = pd.read_csv("tmp.csv")
    data1.drop(columns=['na_item'], inplace=True)
    data2 = pd.melt(data1, id_vars=['nace_r2', 'geo\TIME_PERIOD'])
    data2.columns = ['Sector', 'Region', 'Date', 'Data']

    df = data2[data2['Region'] == country]
    #df = df[~df['Region'].isin(['EA', 'EA12','EA19','EU15', 'EU27_2020', 'EU28'])]

    df['Date'] = pd.PeriodIndex(df.Date, freq='Q').to_timestamp()
    df.set_index("Date", inplace=True)
    df = pd.pivot(df, columns='Sector', values='Data')
    #df = df.dropna()
    df = np.log(df.iloc[:, :])

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
    print(fig.data[0])

    return graphJSON2


def gm_sector(sector='C'):

    data1 = pd.read_csv("tmp.csv")
    data1.drop(columns=['na_item'], inplace=True)
    data2 = pd.melt(data1, id_vars=['nace_r2', 'geo\TIME_PERIOD'])
    data2.columns = ['Sector', 'Region', 'Date', 'Data']

    df = data2[data2['Sector'] == sector]
    df = df[~df['Region'].isin(['EA', 'EA12','EA19','EU15', 'EU27_2020', 'EU28'])]

    df['Date'] = pd.PeriodIndex(df.Date, freq='Q').to_timestamp()
    df.set_index("Date", inplace=True)
    df = pd.pivot(df, columns='Region', values='Data')
    #df = df.dropna()
    df = np.log(df.iloc[:, :])

    labels = {
                 "value": "Natural Log Millions"
             }

    fig = px.line(df, x = df.index, y = df.columns, title= "Sector: " + sector, labels = labels, template="seaborn", width = 1200, height = 1200)
    fig.update_layout(
        autosize= False,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 1)',
        'paper_bgcolor': 'rgba(128, 128, 128, .3)',
    })

    fig.update_yaxes(automargin=True)

    graphJSON1 = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    print(fig.data[0])

    return graphJSON1
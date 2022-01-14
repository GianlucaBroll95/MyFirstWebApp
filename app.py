import numpy as np
from flask import Flask, render_template, request
from flask_caching import Cache
from wtforms import Form, StringField, SubmitField, SelectField
from model_class import Portfolio, TimeSeriesDownloader
import pandas as pd

app = Flask(__name__)
cache = Cache(config={'CACHE_TYPE': 'simple'})
cache.init_app(app)

DEFAULT_TC = 0.02
REB_CHOICES = ["Daily", "Weekly", "Monthly", "Quarterly", "Semi-Annually", "Annually"]
REB_MAP = {"Daily": 1, "Weekly": 5, "Monthly": 22, "Quarterly": 66, "Semi-Annually": 132, "Annually": 264}

HISTORY_CHOICE = ["1 Year", "2 Years", "5 Years", "10 Years", "Max"]
HISTORY_MAP = {"1 Year": 1, "2 Years": 2, "5 Years": 5, "10 Years": 10, "Max": None}


@app.route("/", methods=["GET"])
def home_page():
    return render_template("home_page.html", input_data=InputData())


@app.route("/", methods=["POST"])
def chart_page():
    input_data = InputData(request.form)
    rebalancing_frequency = REB_MAP.get(input_data.rebalancing_frequency.data, None)
    portfolio = Portfolio(data=get_data(), rebalancing_frequency=rebalancing_frequency,
                          t_cost=np.linspace(0, 0.05, 50))
    time_index, gross, net = portfolio.plotting_data()
    return render_template("home_page.html", gross=gross, net=net,
                           labels=time_index, input_data=InputData(),
                           tc_rangevalue=list(np.linspace(0, 0.05, 50).round(3) * 100), show_chart=True)


@cache.cached()
def get_data(start_date=None):
    input_data = InputData(request.form)
    length = HISTORY_MAP.get(input_data.data_length.data, None)
    if length is not None:
        start_date = pd.to_datetime("today") - pd.tseries.offsets.DateOffset(years=length)
    tickers = input_data.tickers.data.split(", ")
    return TimeSeriesDownloader(tickers=tickers, start_date=start_date).download_data()


class InputData(Form):
    tickers = StringField("Tickers:", default="BAC, BF-B, MMM, T")
    # rebalancing_frequency = StringField("Rebalancing Frequency:", default=DEFAULT_REB)
    rebalancing_frequency = SelectField("Rebalancing Frequency:", choices=REB_CHOICES)
    data_length = SelectField("History lenght:", choices=HISTORY_CHOICE)
    button = SubmitField("Get Portfolio")


app.run(debug=True)

import numpy as np
from flask import Flask, render_template, request
from flask_caching import Cache
from wtforms import Form, StringField, SubmitField, SelectField, IntegerField
from wtforms.validators import NumberRange
from model_class import Portfolio, TimeSeriesDownloader
import pandas as pd

app = Flask(__name__)
cache = Cache(config={'CACHE_TYPE': 'simple'})
cache.init_app(app)

REB_CHOICES = [1, 5, 22, 66, 123, 264]
T_COST = np.linspace(0, 0.05, 50)
HISTORY_CHOICE = ["1 Year", "2 Years", "5 Years", "10 Years", "Max"]
HISTORY_MAP = {"1 Year": 1, "2 Years": 2, "5 Years": 5, "10 Years": 10, "Max": None}


@app.route("/", methods=["GET", "POST"])
def home_page():
    if request.method == "GET":
        return render_template("home_page.html", input_data=InputData())
    else:
        input_data = InputData(request.form)
        portfolio = Portfolio(data=get_data(), rebalancing_frequency=REB_CHOICES,
                              t_cost=T_COST, initial_wealth=int(input_data.initial_wealth.data))
        time_index, gross, net = portfolio.plotting_data()
        return render_template("home_page.html", gross=gross, net=net,
                               labels=time_index, input_data=InputData(),
                               tc_rangevalue=list(np.linspace(0, 0.05, 50).round(3) * 100), show_chart=True)


def get_data(start_date=None):
    input_data = InputData(request.form)
    tickers = input_data.tickers.data.split(", ")
    length = HISTORY_MAP.get(input_data.data_length.data, None)
    if length is not None:
        start_date = pd.to_datetime("today") - pd.tseries.offsets.DateOffset(years=length)
    return TimeSeriesDownloader(tickers=tickers, start_date=start_date).download_data()


class InputData(Form):
    tickers = StringField("Tickers:", default="BAC, BF-B, MMM, T")
    data_length = SelectField("History lenght:", choices=HISTORY_CHOICE)
    initial_wealth = IntegerField("Initial Investment:", default=1000,
                                  validators=[NumberRange(min=1000)])
    button = SubmitField("Get Portfolio")


app.run(debug=True)

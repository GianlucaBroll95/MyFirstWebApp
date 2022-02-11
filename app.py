import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from wtforms import Form, StringField, SubmitField, SelectField, IntegerField
from wtforms.validators import NumberRange, DataRequired

from model_class import Portfolio, TimeSeriesDownloader

app = Flask(__name__)

REB_CHOICES = [1, 5, 22, 66, 123, 264]
T_COST = np.linspace(0, 0.05, 50).round(3)
HISTORY_CHOICE = ["1 Year", "2 Years", "5 Years", "10 Years", "Max"]
HISTORY_MAP = {"1 Year": 1, "2 Years": 2, "5 Years": 5, "10 Years": 10, "Max": None}


@app.route("/", methods=["GET", "POST"])
def home_page():
    if request.method == "GET":
        return render_template("home_page.html", input_data=InputData(), show_input=True, show_chart=False)
    else:
        input_data = request.form.to_dict()
        if input_data.get("portfolio_strategy") == "custom":
            if input_data.get("custom_weights") is None:
                raise ValueError("It seems that your custom weight vector is empty")
            strategy = list(map(lambda x: float(x), input_data.get("custom_weights").split(",")))
        else:
            strategy = input_data.get("portfolio_strategy")
        portfolio = Portfolio(data=get_data(),
                              rebalancing_frequency=REB_CHOICES,
                              t_cost=T_COST,
                              initial_wealth=int(input_data.get("initial_wealth")),
                              strategy=strategy)

        time_index, gross, net = portfolio.plotting_data()
        tickers = input_data["tickers"].replace(" ", "").split(",")
        return render_template("home_page.html", gross=gross, net=net,
                               labels=time_index, input_data=InputData(),
                               tc_rangevalue=list(map(lambda x: f"{x:.1%}", T_COST)), show_chart=True, show_input=False,
                               tickers=tickers)


@app.errorhandler(Exception)
def error(err):
    return render_template("error_page.html", e=str(err))


def get_data():
    input_data = InputData(request.form)
    tickers = input_data.tickers.data.replace(" ", "").split(",")
    length = HISTORY_MAP.get(input_data.data_length.data)
    if length is None:
        start_date = None
    elif request.form.to_dict().get("portfolio_strategy") in ["MSR", "GMV"]:
        start_date = pd.to_datetime("today") - pd.tseries.offsets.DateOffset(
            years=length) - pd.offsets.BDay(Portfolio.OFFSET)
    else:
        start_date = pd.to_datetime("today") - pd.tseries.offsets.DateOffset(years=length)
    return TimeSeriesDownloader(tickers=tickers, start_date=start_date).download_data()


class InputData(Form):
    tickers = StringField("Tickers:",
                          default="BAC, BF-B, MMM, T",
                          id="tickers",
                          validators=[DataRequired()])
    data_length = SelectField("History length:",
                              choices=HISTORY_CHOICE,
                              id="history")
    custom_weights = StringField("Custom weights:",
                                 id="custom_weights")
    initial_wealth = IntegerField("Initial Investment ($):",
                                  default=1000,
                                  validators=[NumberRange(min=1000), DataRequired()],
                                  id="initial_wealth")
    button = SubmitField("Get Portfolio")


app.run(debug=True)

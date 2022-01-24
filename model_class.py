"""
Useful classes
"""

import yfinance as yf
import numpy as np
import pandas as pd
from descriptor.descriptors import WeightVector, Tickers, RebalancingFrequency
from fredapi import Fred
from helper_function import minimum_global_variance_weights, tangency_portfolio_weights, calculate_weight_evolution, \
    calculate_portfolio_returns, calculate_portfolio_price, structure_plotting_data
import time


class TimeSeriesDownloader:
    """
    A base class for downloading financial timeseries from Yahoo finance
    """
    tickers = Tickers()

    def __init__(self, tickers, start_date=None, end_date=None, info="Adj Close"):
        """

        Args:
            tickers (list): list of stock tickers to be downloaded
            start_date (str): starting date as %Y-%m-%d
            end_date (str): starting date as %Y-%m-%d
            info (str): type of information downloaded - Open, High, Low, Close or Adj Close
        """

        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.info = info

    def download_data(self):
        print("downloading...")
        try:
            bulk_data = yf.download(tickers=self.tickers, start=self.start_date, end=self.end_date, progress=False)
        except ValueError() as err:
            raise err
        return bulk_data[self.info].dropna()

    def get_company_name(self):
        tickers_name = list()
        for tic in self.tickers:
            tickers_name.append(yf.Tickers(self.tickers).tickers.get(tic).info["longName"])
        return tickers_name


class RiskFreeDownloader:
    """
    Base class for handling risk-free data download from FRED
    """

    def __init__(self):
        self.fred = Fred(api_key="4bf00ae38a403ae4c45a2e775dfa8626")

    def get_data(self, index):
        start = index[0].strftime("%Y-%m")
        end = index[-1].strftime("%Y-%m")
        try:
            data = self.fred.get_series("TB3MS").loc[start:end]
        except ValueError as err:
            try:
                time.sleep(5)
                data = self.fred.get_series("TB3MS").loc[start:end]
            except ValueError as err:
                raise ValueError("Failed to retrieve data for risk-free rate")
        data.index = pd.to_datetime(data.index)
        data.value = data.astype(float)
        data /= (100 * 12)
        return data.reindex(index, method="ffill")


class Portfolio:
    """
    A Base class for portfolio object
    """
    weight = WeightVector(sterilize_attr=["_weights", "_turnover", "_portfolio_price", "_portfolio_return"])
    rebalancing_frequency = RebalancingFrequency(["_weights", "_turnover", "_portfolio_price", "_portfolio_return"])
    OFFSET = 1000

    def __init__(self, data, rebalancing_frequency, strategy="EW", t_cost=0, initial_wealth=1000):
        """
        Args:
            data (pd.DataFrame): pandas dataframe
            rebalancing_frequency (str | pandas.tseries.offsets | int | list): frequency with which the portfolio is
                                                            rebalanced, valid Pandas Offsets subclasses are,
                                                            for example: "B", "W", "BM" (see Pandas Offsets
                                                            documentation for the full list). If an integer is provided,
                                                            it indicates the holding period, in days, between
                                                            rebalancing date.
            strategy (list | string): portfolio stocks weight, default is "EW" (equally weighted), alternative
                                            can be "GMV" (global minimum variance) or "MSR" (max sharpe ratio). If an
                                            array is provided, Portfolio will use it as custom weight vector.
            t_cost (float | np.ndarray): transaction cost of trading stocks
            initial_wealth (int | float): initial amount invested in the portfolio
        """

        self.data = data
        self.rebalancing_frequency = rebalancing_frequency
        self.strategy = strategy
        self.tc = t_cost
        self.initial_wealth = initial_wealth
        # self.ticker_names = TimeSeriesDownloader(list(data.columns)).get_company_name()

        # Caching variables:
        self._weights = None
        self._turnover = None
        self._risk_free = None

    def portfolio_price(self):
        """
        Function that calculate portfolio price
        Returns:
            Pandas DataFrame of portfolio price
        """
        gross_price, net_price = calculate_portfolio_price(**self.portfolio_return(),
                                                           initial_wealth=self.initial_wealth)
        return {"gross_prices": gross_price, "net_prices": net_price}

    def portfolio_return(self):
        """
        This function determines the portfolio historical returns, on daily basis
        Returns:
            Pandas DataFrame of portfolio returns
        """
        gross_ret, net_ret = calculate_portfolio_returns(ret=self._get_ret(),
                                                         **self._weight_evolution(),
                                                         trading_cost=self.tc,
                                                         strategy=self.strategy,
                                                         offset=Portfolio.OFFSET)

        return {"gross_return": gross_ret, "net_return": net_ret}

    def plotting_data(self):
        return structure_plotting_data(**self.portfolio_price())

    def _weight_evolution(self):
        """
        Internal function to calculate the daily portfolio weight evolution and the turnover given the chosen
        rebalance frequency. It is necessary to determine the correct mark-to-market value of the portfolio.
        Returns:
            tuple, Pandas DataFrame of portfolio weight and Pandas DataFrame of turnover
        """
        if self._weights is None and self._turnover is None:
            self._weights, self._turnover = calculate_weight_evolution(ret=self._get_ret(),
                                                                       risk_free=self._risk_free_data(),
                                                                       rebalancing_frequency=self.rebalancing_frequency,
                                                                       rebalancing_dates=self._rebalancing_date(),
                                                                       initial_weights=self._initial_weights(),
                                                                       strategy=self.strategy,
                                                                       offset=Portfolio.OFFSET)
        return {"weights": self._weights, "turnovers": self._turnover}

    def _rebalancing_date(self):
        """
        Internal function to determine the rebalancing date
        Returns:
            Index of rebalancing dates
        """
        if self.strategy == "EW" or isinstance(self.strategy, list):
            ret = self._get_ret()
        else:
            temp = self._get_ret()
            ret = temp.loc[(temp.index[0] + pd.offsets.BDay(Portfolio.OFFSET)):]
        if isinstance(self.rebalancing_frequency, list):
            return [ret.index[np.arange(0, len(ret), reb)] for reb in self.rebalancing_frequency]
        elif isinstance(self.rebalancing_frequency, int):
            return ret.index[np.arange(0, len(ret), self.rebalancing_frequency)]
        else:
            return ret.asfreq(self.rebalancing_frequency).index

    def _get_ret(self):
        """
        Internal function to calculate stock returns
        Returns:
            Pandas DataFrame of simple returns, it also prunes away NA rows.
        """
        return self.data.pct_change(1).dropna()

    def _initial_weights(self):
        """
        Helper function to calculate initial portfolio weights
        Returns:
            np.ndarray of weights
        """
        if isinstance(self.strategy, list):
            return np.array(self.strategy)
        elif self.strategy != "EW":
            full_history = self._get_ret()
            data = full_history.loc[:(full_history.index[0] + pd.offsets.BDay(Portfolio.OFFSET))]
            if self.strategy == "GMV":
                return minimum_global_variance_weights(data)
            else:
                return tangency_portfolio_weights(data, RiskFreeDownloader().get_data(data.index))
        else:
            return np.ones(self.data.shape[1]) / self.data.shape[1]

    def _risk_free_data(self):
        """
        Helper function to get and cache risk-free data
        """
        if self._risk_free is None and self.strategy == "MSR":
            print("downloading risk free")
            self._risk_free = RiskFreeDownloader().get_data(self.data.index)
        return self._risk_free

#
# data = TimeSeriesDownloader(["BAC", "BF-B", "MMM", "T"],
#                             start_date=pd.to_datetime("today") - pd.tseries.offsets.DateOffset(
#                                 years=1)).download_data()
# p = Portfolio(data, rebalancing_frequency=1, strategy="EW", t_cost=0.02)
# print(p.plotting_data())

# TODO: adding portfolio with no rebalancing (one time purchase)

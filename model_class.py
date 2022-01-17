"""
Useful classes
"""
from datetime import datetime

import yfinance as yf
import numpy as np
import pandas as pd
from descriptor.descriptors import WeightVector, Tickers, RebalancingFrequency
from full_fred.fred import Fred
from helper_function import minimum_global_variance_weights, max_sharpe_ratio_weight, update_weight, transaction_cost
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
        return bulk_data[[self.info]].dropna()


class RiskFreeDownloader:
    """
    Base class for handling risk-free data download from FRED
    """

    def __init__(self, api_key="fred_api_key.txt"):
        self.fred = Fred(api_key)

    def get_data(self, index):
        start = index[0].strftime("%Y-%m")
        end = index[-1].strftime("%Y-%m")
        try:
            data = self.fred.get_series_df("TB3MS")[["date", "value"]].set_index("date").loc[start:end]
        except ValueError:
            try:
                time.sleep(5)
                data = self.fred.get_series_df("TB3MS")[["date", "value"]].set_index("date").loc[start:end]
            except ValueError:
                raise ValueError("Failed to retrieve data for risk-free rate")
        data.index = pd.to_datetime(data.index)
        data.value = data.value.astype(float) / 100 / 255
        return data.reindex(index, method="ffill")


class Portfolio:
    """
    A Base class for portfolio object
    """
    weight = WeightVector(sterilize_attr=["_weights", "_turnover", "_portfolio_price", "_portfolio_return"])
    rebalancing_frequency = RebalancingFrequency(["_weights", "_turnover", "_portfolio_price", "_portfolio_return"])
    OFFSET = 365

    def __init__(self, data, rebalancing_frequency, weight="EW", t_cost=0, initial_wealth=1000):
        """
        Args:
            data (pd.DataFrame): pandas dataframe
            rebalancing_frequency (str | pandas.tseries.offsets | int | np.array): frequency with which the portfolio is
                                                            rebalanced, valid Pandas Offsets subclasses are,
                                                            for example: "B", "W", "BM" (see Pandas Offsets
                                                            documentation for the full list). If an integer is provided,
                                                            it indicates the holding period, in days, between
                                                            rebalancing date.
            weight (array | string): portfolio stocks weight, default is "EW" (equally weighted), alternative
                                            can be "GMV" (global minimum variance) or "MSR" (max sharpe ratio). If an
                                            array is provided, Portfolio will use it as custom weight vector.
            t_cost (float | np.array): transaction cost of trading stocks
            initial_wealth (int | flaot): initial amount invested in the portfolio
        """

        self.data = data
        self.rebalancing_frequency = rebalancing_frequency
        self.weight = weight
        self.tc = t_cost
        self.initial_wealth = initial_wealth
        # Caching variables:
        self._weights = None
        self._turnover = None
        self._portfolio_price = None
        self._portfolio_return = None

    def portfolio_price(self):
        """
        Function that calculate portfolio price
        Returns:
            Pandas DataFrame of portfolio price
        """
        if self._portfolio_price is None:
            gross_data, net_data = self.portfolio_return()  # list of tuples - (gross, net)
            if isinstance(self.rebalancing_frequency, list):
                gross_price_matrix = list()
                net_price_matrix = list()
                for gross_ret, net_ret in zip(gross_data, net_data):
                    gross = self.initial_wealth * np.cumprod(
                        1 + gross_ret[["Gross Return"]].rename(columns={"Gross Return": "Gross"}), axis=0)
                    net = self.initial_wealth * np.cumprod(1 + net_ret, axis=0)
                    gross_price_matrix.append(gross)
                    net_price_matrix.append(net)
                self._portfolio_price = gross_price_matrix, net_price_matrix
            else:
                gross = self.initial_wealth * np.cumprod(
                    1 + self.portfolio_return()[0][["Gross Return"]].rename(columns={"Gross Return": "Gross"}), axis=0)
                net = self.initial_wealth * np.cumprod(1 + self.portfolio_return()[1], axis=0)
                self._portfolio_price = gross, net
        return self._portfolio_price

    def portfolio_return(self):
        """
        This function determines the portfolio historical returns, on daily basis
        Returns:
            Pandas DataFrame of portfolio returns
        """

        if self._portfolio_return is None:
            weights, turnover = self._weight_evolution()
            ret = self._get_ret() if self.weight == "EW" else self._get_ret().iloc[Portfolio.OFFSET:]

            if isinstance(self.rebalancing_frequency, list):
                gross_ret_matrix = list()
                net_ret_matrix = list()
                for w, t in zip(weights, turnover):
                    port_ret = np.diagonal(np.matmul(ret.to_numpy(), w.to_numpy().T))
                    trading_cost = transaction_cost(t, self.tc)
                    gross_ret = pd.merge(pd.DataFrame(port_ret, index=ret.index, columns=["Gross Return"]),
                                         trading_cost, left_index=True, right_index=True, how="outer")
                    if self.tc.shape[0] > 1:
                        net_ret = pd.DataFrame(
                            gross_ret[["Gross Return"]].values - gross_ret.iloc[:, 1:].fillna(value=0).values,
                            index=gross_ret.index)
                    else:
                        net_ret = gross_ret["Gross Return"].sub(gross_ret["Transaction Cost"], fill_value=0)
                    gross_ret_matrix.append(gross_ret[["Gross Return"]])
                    net_ret_matrix.append(net_ret)
                self._portfolio_return = gross_ret_matrix, net_ret_matrix
            else:
                port_ret = np.diagonal(np.matmul(ret.to_numpy(), weights.to_numpy().T))
                trading_cost = transaction_cost(turnover, self.tc)
                gross_ret = pd.merge(pd.DataFrame(port_ret, index=ret.index, columns=["Gross Return"]),
                                     trading_cost, left_index=True, right_index=True, how="outer")
                if self.tc.shape[0] > 1:
                    net_ret = pd.DataFrame(
                        gross_ret[["Gross Return"]].values - gross_ret.iloc[:, 1:].fillna(value=0).values,
                        index=gross_ret.index)
                else:
                    net_ret = gross_ret["Gross Return"].sub(gross_ret["Transaction Cost"], fill_value=0)
                self._portfolio_return = gross_ret[["Gross Return"]], net_ret
        return self._portfolio_return

    def plotting_data(self):
        if isinstance(self.rebalancing_frequency, list):
            gross_data, net_nata = self.portfolio_price()
            time_index = gross_data[0].index.strftime("%Y-%m-%d").tolist()
            gross = [data.round(2).values.squeeze().tolist() for data in gross_data]
            net = [[d[1].round(2).values.squeeze().tolist() for d in data.iteritems()] for data in net_nata]
            return time_index, gross, net
        else:
            gross = self.portfolio_price()[0].round().values.squeeze().tolist()
            net = self.portfolio_price()[1].round()
            time_index = self.portfolio_price()[0].index.strftime("%Y-%m-%d").tolist()
            return time_index, gross, [list(d[1]) for d in net.iteritems()]

    def _weight_evolution(self):
        """
        Internal function to calculate the daily portfolio weight evolution and the turnover given the chosen
        rebalance frequency. It is necessary to determine the correct mark-to-market value of the portfolio.
        Returns:
            tuple, Pandas DataFrame of portfolio weight and Pandas DataFrame of turnover
        """
        if self._weights is None and self._turnover is None:
            ret = self._get_ret() if self.weight == "EW" else self._get_ret().iloc[Portfolio.OFFSET:]
            if isinstance(self.rebalancing_frequency, list):
                weight_matrix = list()
                turn_matrix = list()
                rebalancing_date = self._rebalancing_date()
                for idx, reb in enumerate(self.rebalancing_frequency):
                    weights = np.expand_dims(self._initial_weights(), axis=0)
                    turnover = dict()
                    for n in range(1, len(ret)):
                        if (d := ret.index[n - 1]) in rebalancing_date[idx]:
                            w_next = self._calculate_weight(current_date=d)
                            if n - 1 == 0:
                                turnover[d] = np.zeros(self.data.shape[1])
                            else:
                                turnover[d] = abs(w_next - update_weight(ret.loc[d], weights[n - 1]))
                        else:
                            w_next = update_weight(ret.iloc[n - 1], weights[n - 1])
                        weights = np.append(weights, np.expand_dims(w_next, axis=0), axis=0)
                    weight_matrix.append(pd.DataFrame(weights, index=ret.index))
                    turn_matrix.append(pd.DataFrame.from_dict(turnover, orient="index", columns=self.data.columns))
                self._weights = weight_matrix
                self._turnover = turn_matrix

            else:
                weights = np.expand_dims(self._initial_weights(), axis=0)
                rebalancing_date = self._rebalancing_date()
                turnover = dict()
                for n in range(1, len(ret)):
                    if (d := ret.index[n - 1]) in rebalancing_date:
                        w_next = self._calculate_weight(current_date=d)
                        if n - 1 == 0:
                            turnover[d] = np.zeros(self.data.shape[1])
                        else:
                            turnover[d] = abs(w_next - update_weight(ret.loc[d], weights[n - 1]))
                    else:
                        w_next = update_weight(ret.iloc[n - 1], weights[n - 1])
                    weights = np.append(weights, np.expand_dims(w_next, axis=0), axis=0)
                self._weights = pd.DataFrame(weights, index=ret.index)
                self._turnover = pd.DataFrame.from_dict(turnover, orient="index", columns=self.data.columns)
        return self._weights, self._turnover

    def _rebalancing_date(self):
        """
        Internal function to determine the rebalancing date
        Returns:
            Index of rebalancing dates
        """
        ret = self._get_ret().iloc[(0 if self.weight == "EW" else Portfolio.OFFSET):]

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

    def _calculate_weight(self, current_date=None):
        """
        Internal function to calculate portfolio weights at current_date
        Args:
            current_date (str | datetime): date at which perform optimization. It is used to sub-sample the dataframe
             at the correct time
        Returns:
            A numpy array of the portfolio weight vector
        """
        data = self._get_ret().loc[(current_date - pd.offsets.BDay(Portfolio.OFFSET)):current_date]
        if self.weight == "EW":
            return np.ones(self.data.shape[1]) / self.data.shape[1]
        elif self.weight == "GMV":
            return minimum_global_variance_weights(data)
        elif self.weight == "MSR":
            return max_sharpe_ratio_weight(data, RiskFreeDownloader().get_data(data.index))
        else:
            return self.weight

    def _initial_weights(self):
        if self.weight != "EW":
            data = self._get_ret().iloc[:Portfolio.OFFSET]
            if self.weight == "GMV":
                return minimum_global_variance_weights(data)
            else:
                return max_sharpe_ratio_weight(data, RiskFreeDownloader().get_data(data.index))
        else:
            return np.ones(self.data.shape[1]) / self.data.shape[1]

    # def _get_price(self):
    #     """
    #     Internal function to download stock prices from Yahoo.
    #     Returns:
    #         Pandas dataframe
    #     """
    #     if self._price_data is None:
    #         self._price_data = TimeSeriesDownloader(tickers=self.tickers, start_date=self.start_date,
    #                                                 end_date=self.end_date).download_data().dropna()
    #     return self._price_data

#
# data = TimeSeriesDownloader(["BAC", "BF-B", "MMM", "T"]).download_data()
# p = Portfolio(data, rebalancing_frequency=20, weight="EW", t_cost=np.array([0.01, 0.2]))
# print(p.portfolio_return())
# print(p.portfolio_price())
# print(np.array(p.plotting_data()[2]).shape)

from collections.abc import Iterable
import numpy as np
import pandas as pd
from cvxopt import solvers
from cvxopt import matrix

solvers.options["show_progress"] = False


def minimum_global_variance_weights(data):
    """
    Helper function to perform portfolio optimisation
    Args:
        data (pandas.DataFrame): Pandas DataFrame of returns of portfolio stocks
    Returns:
        weight vector for Global Minimum Variance portfolio with short-sale disallowed
    """
    data = data.resample("BM", label="right").apply(lambda x: np.prod(1 + x) - 1).to_numpy()
    mu = np.mean(data, axis=0)
    q = matrix(np.zeros(len(mu)))
    p = matrix((1 / (len(data) - 1)) * np.matmul(data.T, data) - (len(data) / (len(data) - 1)) * np.matmul(mu, mu.T))

    # Equality constraint implementation
    a = matrix(np.ones((1, len(mu))))
    b = matrix(1.0)

    # Inequality constraint implementation
    g = matrix(np.diag(np.ones(len(mu))))
    h = matrix(np.zeros(len(mu)))

    try:
        solution = solvers.qp(p, q, -g, -h, a, b)
    except ValueError:
        try:
            print("Reducing tolerance...")
            solvers.options["reltol"] = 1e-3
            solvers.options["abstol"] = 1e-3
            solution = solvers.qp(p, q, g, h, a, b)
        except ValueError:
            print("Limiting maxiters...")
            solvers.options["maxiters"] = 15
            solution = solvers.qp(p, q, g, h, a, b)
    finally:
        solvers.options["reltol"] = 1e-6
        solvers.options["abstol"] = 1e-7
        solvers.options["maxiters"] = 100
    return np.array(solution["x"]).reshape((1, len(mu))).squeeze()


def tangency_portfolio_weights(data, r_free):
    """
    Helper function to perform portfolio optimisation
    Args:
        data (pandas.DataFrame): Pandas DataFrame of returns of portfolio stocks
        r_free (pandas.DataFrame): Pandas DataFrame for risk-free data

    Returns:
        weight vector for Tangency (max-sharpe ratio) portfolio with short-sale disallowed
    """
    data = data.resample("BM", label="right").apply(lambda x: np.prod(1 + x) - 1)
    r_free = r_free.resample("BM", label="right").first().reindex(data.index).to_numpy().reshape((-1, 1))
    data = data.to_numpy()
    mu = np.mean(data, axis=0)
    q = matrix(np.zeros(len(mu)))
    p = matrix((1 / (len(data) - 1)) * np.matmul(data.T, data) - (len(data) / (len(data) - 1)) * np.matmul(mu, mu.T))
    # Equality constraint implementation
    a = matrix(np.mean(data - r_free, axis=0).reshape((1, 4)))
    b = matrix(1.0)
    # Inequality constraint implementation
    g = matrix(np.diag(np.ones(len(mu))))
    h = matrix(np.zeros(len(mu)))

    try:
        solution = solvers.qp(p, q, -g, -h, a, b)
    except ValueError:
        try:
            print("Reducing tolerance...")
            solvers.options["reltol"] = 1e-3
            solvers.options["abstol"] = 1e-3
            solution = solvers.qp(p, q, g, h, a, b)
        except ValueError:
            print("Limiting maxiters...")
            solvers.options["maxiters"] = 15
            solution = solvers.qp(p, q, g, h, a, b)
    finally:
        solvers.options["reltol"] = 1e-6
        solvers.options["abstol"] = 1e-7
        solvers.options["maxiters"] = 100
    return np.array(solution["x"] / sum(solution["x"])).reshape((1, len(mu))).squeeze()


def update_weight(ret, weight):
    """
    Helper function to update weight vector accordingly to market movements
    Args:
        ret (np.array): return vector
        weight (np.ndarray): weight vector
    Returns:
        updated weight vector
    """
    return (1 + ret) * weight / sum((1 + ret) * weight)


def transaction_cost(turnover, tc):
    """
    Helper function to calculate portfolio transaction cost from turnover
    Args:
        turnover (pandas.DataFrame): Pandas DataFrame of turnover for each stocks in the portfolio
        tc (float | np.ndarray): transaction cost

    Returns:

    """
    if isinstance(tc, np.ndarray):
        time_index = turnover.index
        tc = np.array(tc).reshape((1, -1))
        turnover = turnover.sum(axis=1).to_numpy().reshape((-1, 1))
        return pd.DataFrame(turnover * tc, index=time_index)
    return turnover.sum(axis=1).to_frame(name="Transaction Cost").mul(tc)


def calculate_weight_evolution(ret, risk_free, rebalancing_frequency, rebalancing_dates, initial_weights,
                               strategy, offset):
    """
    Determines weight evolution and turnover
    Args:
        ret (pandas.DataFrame): dataframe of returns
        risk_free (pandas.DataFrame): dataframe of risk-free data
        rebalancing_frequency (list | int | str): rebalancing frequency or iterable of rebalancing frequency
        rebalancing_dates (list | index): list of rebalancing date
        initial_weights (np.ndarray): initial weight vector
        strategy (str | list): type of portfolio strategy (either EW, GMV, MSR or custom)
        offset (int): used to specified length of data used for optimization
    Returns:
        tuple of weights and turnover
    """

    if strategy != "EW" and not isinstance(strategy, list):
        offset_ret = ret.loc[(ret.index[0] + pd.offsets.BDay(offset)):]
    else:
        offset_ret = ret

    if isinstance(rebalancing_frequency, list):
        weight_matrix = list()
        turn_matrix = list()
        for idx, reb in enumerate(rebalancing_frequency):
            weights = np.expand_dims(initial_weights, axis=0)
            turnover = dict()
            for n in range(1, len(offset_ret)):
                if (d := offset_ret.index[n - 1]) in rebalancing_dates[idx]:
                    w_next = calculate_weight(ret, d, risk_free, strategy, offset)
                    if n - 1 == 0:
                        turnover[d] = np.zeros(offset_ret.shape[1])
                    else:
                        turnover[d] = abs(w_next - update_weight(offset_ret.loc[d], weights[n - 1]))
                else:
                    w_next = update_weight(offset_ret.iloc[n - 1], weights[n - 1])
                weights = np.append(weights, np.expand_dims(w_next, axis=0), axis=0)
            weight_matrix.append(pd.DataFrame(weights, index=offset_ret.index))
            turn_matrix.append(pd.DataFrame.from_dict(turnover, orient="index", columns=offset_ret.columns))
        return weight_matrix, turn_matrix
    else:
        weights = np.expand_dims(initial_weights, axis=0)
        turnover = dict()
        for n in range(1, len(offset_ret)):
            if (d := offset_ret.index[n - 1]) in rebalancing_dates:
                w_next = calculate_weight(ret, d, risk_free, strategy, offset)
                if n - 1 == 0:
                    turnover[d] = np.zeros(offset_ret.shape[1])
                else:
                    turnover[d] = abs(w_next - update_weight(offset_ret.loc[d], weights[n - 1]))
            else:
                w_next = update_weight(offset_ret.iloc[n - 1], weights[n - 1])
            weights = np.append(weights, np.expand_dims(w_next, axis=0), axis=0)
        return pd.DataFrame(weights, index=offset_ret.index), pd.DataFrame.from_dict(turnover, orient="index",
                                                                                     columns=offset_ret.columns)


def calculate_weight(ret, current_date, risk_free, strategy, offset):
    """
    Internal function to calculate portfolio weights at current_date
    Args:
        ret (pandas.DataFrame): dataframe of returns
        current_date (str | datetime): date at which perform optimization. It is used to sub-sample the dataframe
         at the correct time
        risk_free (pandas.DataFrame): dataframe of risk-free data
        strategy (str | list): type of portfolio strategy (either EW, GMV, MSR or custom)
        offset (int): used to specified length of data used for optimization

    Returns:
        A numpy array of the portfolio weight vector
    """
    data = ret.loc[(current_date - pd.offsets.BDay(offset)):current_date]
    if strategy == "EW":
        return np.ones(ret.shape[1]) / ret.shape[1]
    elif strategy == "GMV":
        return minimum_global_variance_weights(data)
    elif strategy == "MSR":
        return tangency_portfolio_weights(data, risk_free)
    else:
        return np.array(strategy)


def calculate_portfolio_returns(ret, weights, turnovers, trading_cost, strategy, offset):
    """

    Args:
        ret (pandas.DataFrame): dataframe of returns
        weights (list | pandas.DataFrame): dataframe of weights or list of dataframe of weights
        turnovers (list | pandas.DataFrame): dataframe of turnover or list of dataframe of turnovers
        trading_cost (float | list): trading costs
        strategy (str | list): type of portfolio strategy (either EW, GMV, MSR or custom)
        offset (int): used to specified length of data used for optimization
    Returns:
        tuple of gross return and net returns
    """
    if strategy != "EW" and not isinstance(strategy, list):
        ret = ret.loc[(ret.index[0] + pd.offsets.BDay(offset)):]

    if isinstance(weights, list) and isinstance(turnovers, list):
        gross_ret_matrix = list()
        net_ret_matrix = list()
        for w, t in zip(weights, turnovers):
            port_ret = np.diagonal(np.matmul(ret.to_numpy(), w.to_numpy().T))
            data_ret = pd.merge(left=pd.DataFrame(port_ret, index=ret.index, columns=["Gross Return"]),
                                right=transaction_cost(t, trading_cost),
                                left_index=True,
                                right_index=True,
                                how="outer")
            if isinstance(trading_cost, Iterable):
                net_ret = pd.DataFrame(
                    data_ret[["Gross Return"]].values - data_ret.iloc[:, 1:].fillna(value=0).values,
                    index=data_ret.index)
            else:
                net_ret = data_ret["Gross Return"].sub(data_ret["Transaction Cost"], fill_value=0)
            gross_ret_matrix.append(data_ret[["Gross Return"]])
            net_ret_matrix.append(net_ret)
        return gross_ret_matrix, net_ret_matrix
    else:
        port_ret = np.diagonal(np.matmul(ret.to_numpy(), weights.to_numpy().T))
        trading_cost = transaction_cost(turnovers, trading_cost)
        data_ret = pd.merge(left=pd.DataFrame(port_ret, index=ret.index, columns=["Gross Return"]),
                            right=trading_cost,
                            left_index=True,
                            right_index=True,
                            how="outer")
        if isinstance(trading_cost, Iterable):
            net_ret = pd.DataFrame(
                data_ret[["Gross Return"]].values - data_ret.iloc[:, 1:].fillna(value=0).values,
                index=data_ret.index)
        else:
            net_ret = data_ret["Gross Return"].sub(data_ret["Transaction Cost"], fill_value=0)
        return data_ret[["Gross Return"]], net_ret


def calculate_portfolio_price(gross_return, net_return, initial_wealth):
    """

    Args:
        gross_return (pandas.DataFrame | list): dataframe of gross returns or a list of dataframe of gross returns
        net_return (pandas.DataFrame | list): dataframe of net returns or a list of dataframe of net returns
        initial_wealth (int): initial value invested
    Returns:
        tuple of gross price and net price
    """
    if isinstance(gross_return, list) and isinstance(net_return, list):
        gross_price_matrix = list()
        net_price_matrix = list()
        for g_ret, n_ret in zip(gross_return, net_return):
            gross = initial_wealth * np.cumprod(
                1 + g_ret[["Gross Return"]].rename(columns={"Gross Return": "Gross"}), axis=0)
            net = initial_wealth * np.cumprod(1 + n_ret, axis=0)
            gross_price_matrix.append(gross)
            net_price_matrix.append(net)
        return gross_price_matrix, net_price_matrix
    else:
        gross = initial_wealth * np.cumprod(
            1 + gross_return[["Gross Return"]].rename(columns={"Gross Return": "Gross"}), axis=0)
        net = initial_wealth * np.cumprod(1 + net_return, axis=0)
        return gross, net


def structure_plotting_data(gross_prices, net_prices):
    if isinstance(gross_prices, list) and isinstance(net_prices, list):
        time_index = gross_prices[0].index.strftime("%Y-%m-%d").tolist()
        gross = [data.round(2).values.squeeze().tolist() for data in gross_prices]
        net = [[d[1].round(2).values.squeeze().tolist() for d in data.iteritems()] for data in net_prices]
        return time_index, gross, net
    else:
        time_index = gross_prices.index.strftime("%Y-%m-%d").tolist()
        gross = gross_prices.round(2).values.squeeze().tolist()
        net = net_prices.round(2)
        return time_index, gross, [list(d[1]) for d in net.iteritems()]

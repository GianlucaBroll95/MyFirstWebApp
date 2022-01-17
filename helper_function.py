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
    data = data.to_numpy()
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


def max_sharpe_ratio_weight(data, r_free):
    """
    Helper function to perform portfolio optimisation
    Args:
        data (pandas.DataFrame): Pandas DataFrame of returns of portfolio stocks
        r_free (pandas.DataFrame): Pandas DataFrame for risk-free data

    Returns:
        weight vector for Tangency (max-sharpe ratio) portfolio with short-sale disallowed
    """
    data = data.to_numpy()
    mu = np.mean(data, axis=0)
    q = matrix(np.zeros(len(mu)))
    p = matrix((1 / (len(data) - 1)) * np.matmul(data.T, data) - (len(data) / (len(data) - 1)) * np.matmul(mu, mu.T))

    # Equality constraint implementation
    a = matrix(np.mean(data - r_free.to_numpy(), axis=0).reshape((1, 4)))
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

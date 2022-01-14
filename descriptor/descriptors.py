"""
A useful set to perform data validation
"""
import numpy as np
import pandas as pd


class WeightVector:
    """
    A data-descriptor that ensures the correct input is passed. The alternatives are:
        - "EW": equally-weighted portfolio
        - "GMV": global minimum variance portfolio
        - "MSR": max Sharpe ratio portfolio
        - array: a custom portfolio construction
    """

    def __init__(self, sterilize_attr=None):
        if sterilize_attr is None:
            sterilize_attr = []
        self.sterilize_attr = sterilize_attr

    def __set_name__(self, owner, name):
        self.property_name = name

    def __set__(self, instance, value):
        if isinstance(value, np.ndarray) or isinstance(value, list):
            if sum(value) != 1:
                raise ValueError("Vector weight must sum to one")
            instance.__dict__[self.property_name] = value
        elif value in ["EW", "GMV", "MSR"]:
            instance.__dict__[self.property_name] = value
        else:
            raise ValueError("The strategy should be either 'EW', 'GMS', 'MSR' or a custom vector")
        if self.sterilize_attr:
            for attr in self.sterilize_attr:
                instance.__dict__[attr] = None

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.property_name, None)


class Tickers:
    """
    Data descriptors to validate the correct input for the tickers
    """

    def __init__(self, sterilize_attr=None):
        if sterilize_attr is None:
            sterilize_attr = []
        self.sterilize_attr = sterilize_attr

    def __set_name__(self, owner, name):
        self.property_name = name

    def __set__(self, instance, value):
        if isinstance(value, list):
            if all(isinstance(tic, str) for tic in value):
                instance.__dict__[self.property_name] = value
            else:
                raise TypeError("Tickers must be a list of string")
        else:
            raise TypeError("Tickers must be a list")
        if self.sterilize_attr:
            for attr in self.sterilize_attr:
                instance.__dict__[attr] = None

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.property_name, None)


class RebalancingFrequency:
    """
    Data descriptors to validate the correct input for the rebalancing frequency
    """
    def __init__(self, sterilize_attr=None):
        if sterilize_attr is None:
            sterilize_attr = []
        self.sterilize_attr = sterilize_attr

    def __set_name__(self, owner, name):
        self.property_name = name

    def __set__(self, instance, value):
        if isinstance(value, int) and value > 0:
            if isinstance(value, bool):
                raise TypeError(
                    f"Invalid frequency type, {self.property_name} must be a valid frequency string or a positive"
                    f" integer")
            instance.__dict__[self.property_name] = value
        elif isinstance(value, str) and value in pd.tseries.frequencies.__dict__["_offset_to_period_map"].keys():
            instance.__dict__[self.property_name] = value
        else:
            raise TypeError(
                f"Invalid frequency type, {self.property_name} must be a valid frequency string or a positive integer")
        if self.sterilize_attr:
            for attr in self.sterilize_attr:
                instance.__dict__[attr] = None

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.property_name, None)

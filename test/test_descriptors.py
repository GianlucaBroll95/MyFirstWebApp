"""
Pytest for descriptors
"""

from descriptor.descriptors import WeightVector, Tickers, RebalancingFrequency
import pytest


@pytest.fixture
def instance_class():
    obj = type("TestClass", (), {"weight": WeightVector(),
                                 "tickers": Tickers(),
                                 "reb_frequency": RebalancingFrequency()})
    return obj()


@pytest.mark.parametrize("weight", [[0.1, 0.2, 0.3, 0.4], "EW", "GMV", "MSR"])
def test_valid_weight(instance_class, weight):
    instance_class.weight = weight
    assert instance_class.weight == weight


@pytest.mark.parametrize("weight", [[0.1, 0.2, 0.3, 0.1], "S", 1, True])
def test_invalid_weight(instance_class, weight):
    with pytest.raises(ValueError):
        instance_class.weight = weight


@pytest.mark.parametrize("tickers", [["AAPL", "MSFT", 1, True], "AAPL, MSFT", 2])
def test_invalid_tickers(instance_class, tickers):
    with pytest.raises(TypeError):
        instance_class.tickers = tickers


def test_valid_tickers(instance_class):
    instance_class.tickers = ["AAPL", "MSFT", "GOOGL"]
    assert instance_class.tickers == ["AAPL", "MSFT", "GOOGL"]


@pytest.mark.parametrize("reb_frequency", [-1, "MONTH", "12", True, ["BM", 12]])
def test_invalid_rebalancing_frequency(instance_class, reb_frequency):
    with pytest.raises(TypeError):
        instance_class.reb_frequency = reb_frequency


@pytest.mark.parametrize("reb_frequency", [30, "BM"])
def test_valid_rebalancing_frequency(instance_class, reb_frequency):
    instance_class.reb_frequency = reb_frequency
    assert instance_class.reb_frequency == reb_frequency

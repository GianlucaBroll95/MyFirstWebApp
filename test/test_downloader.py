"""
Pytest for downloader
"""

import pytest
import pandas as pd
from model_class import TimeSeriesDownloader


@pytest.fixture
def input_data():
    return {
        "tickers": ["AAPL", "MSFT", "AMZN", "GOOGL"],
        "start_date": "2000-01-03",
        "end_date": "2021-12-23",
    }


@pytest.fixture
def downloader(input_data):
    return TimeSeriesDownloader(**input_data)


def test_downloader(downloader, input_data):
    data = downloader.download_data()
    assert sorted(data.columns.tolist()) == sorted(input_data["tickers"])
    assert len(data) > 0
    assert data.index[0].strftime("%Y-%m-%d") == input_data["start_date"]
    assert (data.index[-1] + pd.offsets.Day(1)).strftime("%Y-%m-%d") == input_data["end_date"]

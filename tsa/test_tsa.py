import pytest
from datetime import datetime
import pandas as pd
import numpy as np
from visualization import *
from interpolation import interpolate_dates as interpolate
import properties as props
import sys

# specifying data
df = pd.read_csv("datasets/2/Electric_Production.csv")

# conditions to filter dataframe
filter_conditions = {}
for col, val in filter_conditions.items():
    df = pd.DataFrame(df[df[col] == val])

# columns to drop
drop_cols = []
df = df.drop(drop_cols, axis = 1) if len(drop_cols) else df

#specifying time axis
x = "DATE"
y = df.columns.tolist()
y.remove(x)

# specifying frequency of data
# freq = pd.to_timedelta(np.diff(df[x]).min())

def get_freq(x):
    return 'M' if x.days == 28 else x

def test_input_formats():
    assert (isinstance(df, pd.DataFrame)) and isinstance(x, str) and isinstance (y, list) and isinstance(y[0], str)

def test_continuous_timeseries():
    df[x] = pd.to_datetime(df[x])
    freq = pd.to_timedelta(np.diff(df[x]).min())
    freq = get_freq(freq)
    assert len(interpolate(df, x, y, interval=freq)) == len(df) 

def test_does_not_have_nan():
    assert sum(df[y].isna().sum().values) == 0

def test_time_format_valid():
    try:
        df[x] = pd.to_datetime(df[x])
    except pd._libs.tslibs.parsing.DateParseError:
        pytest.fail("Invalid date format")

def test_trend_working():
    t = props.Trend(df, y[0], x)
    assert t.detrend()["Detrend"].isna().sum() == 0
    
def test_seasonal_working():
    try:
        s = props.Seasonality(df, y[0], x)
        seasonal_component = s.seasonal()
    except ValueError:
        pytest.fail("Problem with frequency or time index.")




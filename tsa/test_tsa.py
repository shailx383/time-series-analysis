import pytest
from datetime import datetime
import pandas as pd
from visualization import *
from interpolation import interpolate_dates as interpolate
import properties as props
import sys

# specifying data
df = pd.read_csv("datasets/1/train.csv")

# conditions to filter dataframe
filter_conditions = {"product":0, "store":0}
for col, val in filter_conditions.items():
    df = pd.DataFrame(df[df[col] == val])

# columns to drop
drop_cols = ['product', 'store']
df = df.drop(drop_cols, axis = 1) if len(drop_cols) else df

#specifying time axis
x = "Date"
y = df.columns.tolist()
y.remove(x)

# specifying frequency of data
freq = 'd'

def test_input_formats():
    assert (isinstance(df, pd.DataFrame)) and (x, str) and (y, list[str])

def test_continuous_timeseries():
    df[x] = pd.to_datetime(df[x])
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


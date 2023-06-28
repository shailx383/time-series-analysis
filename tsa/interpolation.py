import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def interpolate_dates(df: pd.DataFrame, x: str, y: list[str], interval = 'D', method = 'linear'):
    '''
        interpolates breaks in discontinuous time series and returns continuous dataframe
        
        Args:
            - df (pd.DataFrame): dataframe of discontinuous time series
            - x (str): column name of datetime axis
            - y (list[str]): list of column names of time series data
            - interval (Any): interval as 'D', 'M', 'A', etc. or datetime.timedelta object
            - method (str): method of interpolation. defaults to linear
            
        Returns:
            - (pd.DataFrame): interpolated, continuous dataframe
    '''
    final_df = pd.DataFrame()
    for column in y:
        idf = pd.DataFrame(df[[x, column]]).set_index([x])
        interpolated = idf.resample(interval).interpolate(method = method)
        final_df = pd.concat([final_df, interpolated], axis=1)
        # print(interpolated)
    final_df[x] = final_df.index
    final_df.index = range(len(final_df))

    return final_df

def _round_dt(dt, delta):
    return datetime.min + round((dt - datetime.min) / delta) * delta

def impute_time_axis(time_axis: pd.Series, freq: timedelta = timedelta(days = 1)):
    '''
        fills null values in the time axis
        
        Args:
            - time_axis (pd.Series): column with datetime info
            - freq (timedelta): frequency of time series
            
        Returns:
            - (pd.Series): datetime column with no null values
    '''
    if time_axis.isna().sum() == 0:
        return pd.to_datetime(time_axis)
    series = pd.to_datetime(time_axis)
    print(series[0].minute)
    time_list = series.to_list()
    null_intervals = []
    idx = 0
    while idx < len(time_list):
        if pd.isnull(time_list[idx]):
            start = time_list[idx-1]
            nulls = 0
            while pd.isnull(time_list[idx + nulls]):
                nulls += 1
            idx += nulls
            end = time_list[idx]
            null_intervals.append({
                "start": start,
                "end": end,
                "nulls": nulls + 2
            })
        else:
            idx += 1
    seconds_intervals =  [np.linspace(interval["start"].timestamp(), interval["end"].timestamp(), interval["nulls"]) for interval in null_intervals]
    x = []
    for i in seconds_intervals:
        for j in i[1:-1]:
            x.append(j)
    final_fill_values = [pd.to_datetime(_round_dt(datetime.utcfromtimestamp(i), freq)) for i in x]
    column = []
    idx = 0
    for i in time_axis.to_list():
        if pd.isnull(i):
            column.append(final_fill_values[idx])
            idx += 1
        else:
            column.append(i)
    ser = pd.Series(pd.to_datetime(column))
    return ser
    
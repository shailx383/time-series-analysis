import pandas as pd

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
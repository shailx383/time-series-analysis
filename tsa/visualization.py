import pandas as pd
from sklearn.cluster import KMeans

def single_line_plot(df: pd.DataFrame, x: str, y: str, remove_time: bool = False):
    '''
        returns an object with data which is used to plot Apache E Charts waterfall plot
        
        Args:
            - df (pd.DataFrame): dataframe with time series to be plotted
            - x (str): column name of datetime axis
            - y (str): column name of time series data
            - title (str): title of plot
            - remove_time (bool): True will not include time with the date strings in the output
    '''
    x_plot =  [str(i.date())+' '+str(i.time()) for i in df[x].to_list()] if not remove_time else [str(i.date()) for i in df[x].to_list()]
    return {
    'title': 'Plot of '+ y + ' over '+x+':',
    'y': df[y].values.tolist(),
    'x': x_plot
    }
    
def double_line_plot(df: pd.DataFrame, x: str, y1: str, y2: str, remove_time: bool = False):
    '''
        returns an object with data which is used to plot Apache E Charts waterfall plot
        
        Args:
            - df (pd.DataFrame): dataframe with time series to be plotted
            - x (str): column name of datetime axis
            - y1 (str): first column name of time series data
            - y2 (str): second column name of time series data
            - title (str): title of plot
            - remove_time (bool): True will not include time with the date strings in the output
    '''
    x_plot =  [str(i.date())+' '+str(i.time()) for i in df[x].to_list()] if not remove_time else [str(i.date()) for i in df[x].to_list()]
    return {
    'title': 'Plot of '+ y1 +' and ' +y2 + ' over '+x+':',
    'y1': df[y1].values.tolist(),
    'y2': df[y2].values.tolist(),
    'x': x_plot
    }

def waterfall_plot(df: pd.DataFrame, x: str, y: str, title: str, remove_time: bool = False):
    '''
        returns an object with data which is used to plot Apache E Charts waterfall plot
        
        Args:
            - df (pd.DataFrame): dataframe with time series to be plotted
            - x (str): column name of datetime axis
            - y (str): column name of time series data
            - title (str): title of plot
            - remove_time (bool): True will not include time with the date strings in the output
        
        Returns:
            -(dict):
                - 't': title of the plot
                - 'x': x-axis data
                - 'y': y-axis data (starting points of bars)
                - 'i': ending points of increasing bars
                - 'd': ending points of decreasing bars 
    '''
    vals = df[y].to_list()
    color_labels =[]
    for i in range (1, len(vals)):
        if vals[i] > vals[i-1]:
            color_labels.append('i')
            i_list = [vals[0]]
            d_list = ['-']
        else:
            color_labels.append('d')
            i_list = ['-']
            d_list = [0]
    y = [0]
    c = 0
    for i in range(1, len(vals)):
        if color_labels[c] == 'd':
            y.append(round(vals[i], 2))
            d_list.append(round(vals[i-1] - vals[i], 2))
            i_list.append('-')
        else:
            y.append(vals[i-1])
            i_list.append(round(vals[i] - vals[i-1], 2))
            d_list.append('-')
        c += 1
    
    x_plot =  [str(i.date())+' '+str(i.time()) for i in df[x].to_list()] if not remove_time else [str(i.date()) for i in df[x].to_list()]
    return {
        't': title,
        "x": x_plot, 
        'y': y,
        'i': i_list,
        'd': d_list
    }
    
def stacked_area_plot(df: pd.DataFrame, columns: list[str], x: str, title: str, remove_time: bool = False):
    '''
        returns object which can be used to plot Apache E Charts stacked area chart
        
        Args: 
            - df (pd.DataFrame): dataframe with the time series columns
            - columns (list[str]): list of names of columns to be plotted in area chart
            - x (str): name of datetime axis column
            - title (str): title of plot
            - remove_time (bool): True will not include time with the date strings in the output
            
        Returns:
            - (dict):
                - 'title': title of plot
                - 'names': names of columns
                - 'vals': data to be plotted
                - 'x': x-axis data
    '''
    x_plot =  [str(i.date())+' '+str(i.time()) for i in df[x].to_list()] if not remove_time else [str(i.date()) for i in df[x].to_list()]
    return {
        'title': title,
        'names': columns,
        'vals': df[columns].values.T.tolist(),
        "x": x_plot
    }

def lag_plot(timeseries: pd.Series, lag: int, title: str, data_reduce: int = 1):
    '''
        returns object for plotting lag plot or scatter plot in Apache Echarts
        
        Args:
            - timeseries (pd.Series): timeseries to be plotted
            - lag (int): lag at which plot is to be made
            - data_reduce (int): factor by which we want to make the lagplot more sparse
            - title (str): title of plot
        
        Returns:
            - (dict) 
                - "data": data to be plotted
                - "title: title of plot
    '''
    x = timeseries.values[:-lag].tolist()
    y = timeseries.values[lag:].tolist()
    data = []
    for i in range(len(x)):
        data.append([x[i], y[i]])
    kmeans = KMeans(int(len(data)/data_reduce))
    kmeans.fit(data)
    data = kmeans.cluster_centers_
    return {"data": data.tolist(),
            "title": title}
    
def stream_graph(df: pd.DataFrame, columns: list[str], x: str, title: str, remove_time: bool= False):
    '''
        returns object which can be used to plot Apache E Charts stacked area chart
        
        Args: 
            - df (pd.DataFrame): dataframe with the time series columns
            - columns (list[str]): list of names of columns to be plotted in area chart
            - x (str): name of datetime axis column
            - title (str): title of plot
            - remove_time (bool): True will not include time with the date strings in the output
            
        Returns:
            - (dict):
                - 'title': title of plot
                - 'names': names of columns
                - 'data': data to be plotted
    '''
    vals = []
    x_plot =  [str(i.date())+' '+str(i.time()) for i in df[x].to_list()] if not remove_time else [str(i.date()) for i in df[x].to_list()]
    for i in columns:
        for val_idx in range(len(df[i].values.tolist())):
            vals.append([x_plot[val_idx], df[i].values[val_idx], i])
    return {
        'data': vals,
        'names': columns,
        'title': title
    }
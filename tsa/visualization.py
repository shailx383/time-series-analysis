import pandas as pd
from sklearn.cluster import KMeans

def waterfall_plot(df: pd.DataFrame, x: str, y: str, title: str):
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
    return {
        't': title,
        "x": [str(i.date())+' '+str(i.time()) for i in df[x].to_list()],
        'y': y,
        'i': i_list,
        'd': d_list
    }
    
def stacked_area_plot(df, columns, x, title):
    return {
        'title': title,
        'names': columns,
        'vals': df[columns].values.T.tolist(),
        "x": [str(i.date())+' '+str(i.time()) for i in df[x].to_list()]
    }

def lag_plot(timeseries, lag, data_reduce = 1):
    x = timeseries.values[:-lag].tolist()
    y = timeseries.values[lag:].tolist()
    data = []
    for i in range(len(x)):
        data.append([x[i], y[i]])
    kmeans = KMeans(int(len(data)/data_reduce))
    kmeans.fit(data)
    data = kmeans.cluster_centers_
    return data.tolist()
    
def stream_graph(df, columns, x):
    vals = []
    for i in columns:
        for val_idx in range(len(df[i].values.tolist())):
            vals.append([[str(i.date()) +' '+str(i.time()) for i in df[x].to_list()][val_idx], df[i].values[val_idx], i])
    return {
        'data': vals,
        'names': columns
    }
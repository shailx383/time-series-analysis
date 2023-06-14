def lag_plot(timeseries, lag):
    x = timeseries.values[:-lag].tolist()
    y = timeseries.values[lag:].tolist()
    data = []
    for i in range(len(x)):
        data.append([x[i], y[i]])
    return data
    
    
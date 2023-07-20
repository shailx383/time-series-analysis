# Time Series Analysis

Plots, visualizations, models, tests and properties of time series

## Data Preprocessing

| No. | Data Type | Name              | Description                                                                                                               |
| --- | --------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------- |
| 1   | Datetime  | Fill Null Values  | Find appropriate datetime objects to fill null values rounded off to the nearest time unit (months, days, hours, minutes) |
| 2   | Numeric   | Fill Null Values  | Use linear interpolation to fill null values in time series                                                               |
| 3   | Numeric   | Detrend           | Remove the trend component of any time series                                                                             |
| 4   | Numeric   | Deseasonalize     | Remove the seasonal component of any time series                                                                          |
| 5   | Numeric   | Make stationary   | Make the time series stationary using rolling/log/power transforms                                                       |
| 5   | Datetime  | Format converison | Convert datetime into a unified format                                                                                    |

## Metrics

| No. | Name                                      | Variable type | Description                                                                        |
| --- | ----------------------------------------- | ------------- | ---------------------------------------------------------------------------------- |
| 1   | Augmented Dickey-Fuller test              | Univariate    | test stationarity of the time series (Null Hyp. series is non-stationary)          |
| 2   | Kwiatkowski–Phillips–Schmidt–Shin test | Univariate    | test stationarity of the time series (Null Hyp. series is stationary)             |
| 3   | Range Unit Root test                      | Univariate    | test stationarity of the time series (Null Hyp. series is non-stationary)          |
| 4   | Durbin-Watson test                        | Univariate    | tests nature of autocorrelation (positive/negative/none)                           |
| 5   | Cross correlation                         | Bivariate     | Value of cross correlation between 2 time series at different lags                 |
| 6   | Granger Causality                         | Bivariate     | Statistic which determines degree to which a series causes another                 |
| 7   | Autocovariance                            | Univariate    | describe the relationship between a time series and its lagged versions            |
| 8   | Cross covariance                          | Bivariate     | describe the relationship between two different time series at different time lags |

## Plots

| No. | Name                    | Variable Type | Description                                                                           |
| --- | ----------------------- | ------------- | ------------------------------------------------------------------------------------- |
| 1   | Single Line Plot        | Univariate    | plot the time series                                                                  |
| 2   | Double Line Plot        | Bivariate     | plot one time series against another                                                  |
| 3   | Waterfall Chart         | Univariate    | display the cumulative effect of sequentially introduced positive and negative values |
| 4   | Lag plot                | Univariate    | plot of a time series against its lagged values                                       |
| 5   | Stacked Area Chart      | Multivariate  | represent multiple series of data as overlapping areas                                |
| 6   | Stream Graph            | Multivariate  | display the changes in the composition and the flow of multiple categories over time  |
| 7   | Trend plot              | Univariate    | plot the trend component of the time series                                           |
| 8   | Detrend plot            | Univariate    | plot time series against detrended version                                            |
| 9   | Seasonal                | Univariate    | plot seasonal component of time series                                                |
| 10  | Deseasonalize           | Univariate    | plot time series against deseasonalized version                                       |
| 11  | Stationary transform    | Univariate    | plot stationary transformed time series                                               |
| 12  | Autocorrelation         | Univariate    | plot ACF graph                                                                        |
| 13  | Partial Autocorrelation | Univariate    | plot PACF graph                                                                       |
| 14  | Autocovariance          | Univariate    | plot ACOVF graph                                                                      |
| 15  | Cross covariance        | Bivariate     | plot CCOVF graph                                                                      |
| 17  | ARIMA forecast plot     | Univariate    | plot forecast given by ARIMA                                                          |

## Models

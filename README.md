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
| 1   | Augmented Dickey-Fuller test              | Univariate    | tests stationarity of the time series (Null Hyp. series is non-stationary)         |
| 2   | Kwiatkowski–Phillips–Schmidt–Shin test | Univariate    | tests stationarity of the time series (Null Hyp. series is stationary)             |
| 3   | Range Unit Root test                      | Univariate    | tests stationarity of the time series (Null Hyp. series is non-stationary)         |
| 4   | Durbin-Watson test                        | Univariate    | tests nature of autocorrelation (positive/negative/none)                           |
| 5   | Cross correlation                         | Bivariate     | Value of cross correlation between 2 time series at different lags                 |
| 6   | Granger Causality                         | Bivariate     | Statistic which determines degree to which a series causes another                 |
| 7   | Autocovariance                            | Univariate    | describe the relationship between a time series and its lagged versions            |
| 8   | Cross covariance                          | Bivariate     | describe the relationship between two different time series at different time lags |

## Plots

| No. | Name               | Variable Type | Description                                                                           |
| --- | ------------------ | ------------- | ------------------------------------------------------------------------------------- |
| 1   | Waterfall Chart    | Univariate    | display the cumulative effect of sequentially introduced positive and negative values |
| 2   | Lag plot           | Univariate    | plot of a time series against its lagged values                                       |
| 3   | Stacked Area Chart | Multivariate  | represents multiple series of data as overlapping area                                |
| 4   | Stream Graph       | Multivariate  | displays the changes in the composition and the flow of multiple categories over time |
| 5   |                    |               |                                                                                       |
| 6   |                    |               |                                                                                       |

## Models

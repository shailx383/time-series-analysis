import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss, range_unit_root_test
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from tsa.dateconverter import DateConverter
from statsmodels.tsa.stattools import acf, pacf

'''
class ExponentialSmoothingModel:
  def __init__(self, df, x, y):
    self.df = df
    self.x = x
    self.y = y
    self.idf = self._idf()
    self.fitted = False

  def _idf(self):
    idf = pd.DataFrame()
    idf[self.y] = self.df[self.y]
    idf[self.x] = self.df[self.x]
    idf = idf.set_index([self.x])
    return idf

  def fit_simple_smoothing(self, smoothing_level = 0.3):
    model = SimpleExpSmoothing(self.idf, initialization_method = 'heuristic')
    fit = model.fit(smoothing_level = smoothing_level, optimized = False)
    self.fit = fit
    self.method = 'simple'
    print("Data has been fit.")
    self.fitted = True

  def fit_holts(self, smoothing_level = 0.8, smoothing_trend=0.2, method = 'linear'):
    self.method = method
    if method == 'linear':
      self.fit = Holt(self.idf, initialization_method="estimated").fit(smoothing_level = smoothing_level, smoothing_trend = smoothing_trend, optimized = False)
    elif method == 'exponential':
      self.fit = Holt(self.idf, exponential = True, initialization_method="estimated").fit(smoothing_level = smoothing_level, smoothing_trend = smoothing_trend, optimized = False)
    elif method == 'damped':
      self.fit = Holt(self.idf, damped_trend = True, initialization_method="estimated").fit(smoothing_level = smoothing_level, smoothing_trend = smoothing_trend, optimized = False)
    else:
      print("Incorrect method for Holt's.")
    self.fitted = True

  def get_forecasts(self, num_periods):
    if not self.fitted:
      print("Fit data first.")
      return
    fcast = self.fit.forecast(num_periods).rename(self.method + " forecast")
    self.fcast = fcast
    return fcast

class MovingAverage:
    def __init__(self, rolling_window : int = 12):
        self.window = rolling_window

    def load_timeseries(self, df: pd.DataFrame, x: str, y: str):
        df = pd.DataFrame(df[x, y])
        df = df.set_index([x])
        self.df = df

    def moving_averages(self):
        self.ma = self.df[self.df.columns[0]].rolling(window = self.window).mean().shift(1).dropna()
        return self.ma

    def plot_moving_average(self):
        self.ma = self.moving_averages()
        plt.figure(figsize=(10,4))
        self.df.plot()
        self.ma.plot()
        plt.legend()

    def _get_mape(self, actual, predicted):
        y_true, y_pred = np.array(actual), np.array(predicted)
        return np.round(np.mean(np.abs((actual-predicted)/actual))*100, )

    def build_model(self, order = 1, show_summary = True):
        arima = ARIMA(self.df, order = (0,0,order))
        ma_model = arima.fit()
        if show_summary:
            ma_model.summary()
        self.model = ma_model

    def mape(self, actual, predicted):
        print(self._get_mape(actual, predicted))

    def get_forecasts(self, num_periods):
        return self.model.forecast(num_periods)
'''
class Arima:
    def __init__(self):
        self._auto = None
        return
    
    def _index_df(self, df: pd.DataFrame, x: str, y: str):
        idf = pd.DataFrame(df[[x, y]])
        converter = DateConverter()
        idf[x] = idf[x].apply(lambda x: converter.convert_date(x, infer_format=True))
        idf = idf.set_index([x])
        return idf
    
    def __call__(self, df: pd.DataFrame, x: str, y: str):
        self._y = y
        self._df = self._index_df(df, x, y)
    
    def fit(self, order = None, show_summary=True):
        if order is None:
            self._auto_arima = pm.auto_arima(self._df, stepwise = False, seasonal=False)
            if show_summary:
                print(self._auto_arima.summary())
            self._auto = True
        else:
            self._auto = False
            model = ARIMA(self._df, order = order)
            self._arima = model.fit()
            if show_summary:
                print(self._arima.summary())
                
    def predict(self, num_periods: int):
        if self._auto is None:
            print("run fit() first")
        elif self._auto is True:
            self._forecasts = self._auto_arima.predict(n_periods = num_periods)
            return self._forecasts
        else:
            self._forecasts = self._arima.forecast(num_periods)
            return self._forecasts
         
    def plot_forecast(self, test_df, y, exclude_time = False):
        if self._auto is None:
            print("run fit() first")
        else:
            forecasts = self.predict(len(test_df))
            if exclude_time:
                x_axis_data = [str(i.date()) for i in self._df.index.tolist()]+[str(i.date()) for i in forecasts.index.tolist()]
            else:
                x_axis_data = [str(i.date()) + ' ' + str(i.time()) for i in self._df.index.tolist()]+[str(i.date()) + ' ' + str(i.time()) for i in forecasts.index.tolist()]
            y_axis_series = self._df[self._df.columns[0]].to_list() + ['-']*len(test_df)
            y_axis_forecast = ['-']*len(self._df) + forecasts.values.tolist()
            y_axis_test = ['-']*len(self._df) + test_df[y].to_list()
            
            return {
                'title': f'Forecast of {self._y}:',
                'x': x_axis_data,
                'y_series': y_axis_series,
                'y_test': y_axis_test,
                'y_forecast': y_axis_forecast
            }
    
    def test_stationarity(self, order_of_differencing = 0,  test = 'adf'):
        
        df = self._df
        
        for i in range(order_of_differencing):
            df = df.diff().dropna()
        
        if test == 'adf':
            dftest = adfuller(df[self._y], autolag = 'AIC')
            print("1. ADF : ",dftest[0])
            print("2. P-Value : ", dftest[1])
            print("3. Num Of Lags : ", dftest[2])
            print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
            print("5. Critical Values :")
            for key, val in dftest[4].items():
              print("\t",key, ": ", val)
            if dftest[1] > 0.05:
                print("\n\nAs p-value is outside the confidence interval of 95%, series is non-stationary.")
            else:
                print("\n\nAs p-value is inside the confidence interval of 95%, series is stationary.")
        
        elif test == 'kpss':
            dftest = kpss(df[self._y], regression = 'ct')
            print("1. KPSS : ",dftest[0])
            print("2. P-Value : ", dftest[1])   
            print("3. Num Of Lags : ", dftest[2])
            print("4. Critical Values :")
            for key, val in dftest[3].items():
                print("\t",key, ": ", val)
            if dftest[1] > 0.05:
                print("\n\nAs p-value is outside the confidence interval of 95%, series is non-stationary.")
            else:
                print("\n\nAs p-value is inside the confidence interval of 95%, series is stationary.")
        elif test == 'rur':
            dftest = range_unit_root_test(df[self._y])
            print("1. RUR Stat : ",dftest[0])
            print("2. P-Value : ", dftest[1])
            print("3. Critical Values :")
            for key, val in dftest[2].items():
                print("\t",key, ": ", val)
            if dftest[1] > 0.05:
                    print("\n\nAs p-value is outside the confidence interval of 95%, series is non-stationary.")
            else:
                    print("\n\nAs p-value is inside the confidence interval of 95%, series is stationary.")
        else:
            raise ValueError("Invalid test name.")
    
    def acf_plot(self, order_of_differencing = 0):
        
        df = self._df
        
        for i in range(order_of_differencing):
            df = df.diff().dropna()
            
        acf_vals, confint = acf(df, alpha = 0.05)
        return {
            'title': f'Autocorrelation plot of {self._y}, order = {order_of_differencing}',
            'y': acf_vals.tolist(),
            'x': list(range(len(acf_vals))),
            'upper': (confint[:, 1] - acf_vals).tolist(),
            'lower': (confint[:, 0] - acf_vals).tolist()
        }
        
    def pacf_plot(self, order_of_differencing = 0):
        df = self._df
        
        for i in range(order_of_differencing):
            df = df.diff().dropna()
            
        pacf_vals, confint = pacf(df, alpha = 0.05)
        return {
            'title': f'Partial Autocorrelation plot of {self._y}, order = {order_of_differencing}',
            'y': pacf_vals.tolist(),
            'x': list(range(len(pacf_vals))),
            'upper': (confint[:, 1] - pacf_vals).tolist(),
            'lower': (confint[:, 0] - pacf_vals).tolist()
        }  
    
    def error_metrics(self, test_df, y):
        forecasts = self.predict(len(test_df))
        return pd.DataFrame([
            
            ["MAPE", mean_absolute_percentage_error(test_df[y], forecasts)],
            ["MAE", mean_absolute_error(test_df[y], forecasts)],
            ["MSE", mean_squared_error(test_df[y], forecasts)]], columns=["Metric", "Value"]
        )
    
        
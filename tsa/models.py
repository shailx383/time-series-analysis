import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima

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

class ModelARIMA:
    def __init__(self):
        self.df = None
        return

    def load_timeseries(self, df: pd.DataFrame, x: str, y: str):
        self.x = x
        self.y = y
        df = pd.DataFrame(df[x, y])
        df = df.set_index([x])
        self.df = df

    def build_model(self, order = None, show_summary = True):
        if order is None:
            self.auto = True
            model = auto_arima(self.df, start_p=1, start_q=1,
                      test='adf',
                      max_p=5, max_q=5,
                      m=1,
                      d=None,
                      seasonal=False,
                      start_P=0,
                      D=0,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)
            self.model_fit = model
            if show_summary:
                print(model.summary())
        else:
            self.auto = False
            model = ARIMA(self.df, order=order)
            model_fit = model.fit()
            if show_summary:
                print(model_fit.summary())
            self.model_fit = model_fit
            print('Model built successfully.')

    def plot_forecast(self, periods):
        if not self.auto:
            fig = plt.figure()
            self.df.plot(label = 'Series')
            fc = self.model_fit.forecast(periods)
            fc.plot(label = 'Forecast')
        else:
            fig = plt.figure()
            self.df.plot(label = 'Series')
            fc = self.model_fit.predict(periods)
            fc.plot(label = 'Forecast')

    def get_forecasts(self, periods):
        if not self.auto:
            return self.model_fit.forecast(periods)
        else:
            return self.model_fit.predict(periods)

    def error_metrics(self, test_series):
        actual = test_series[test_series.columns[0]].to_numpy()
        forecast = self.get_forecasts(len(actual)).values
        mape = np.mean(np.abs(forecast - actual)/np.abs(actual))
        me = np.mean(forecast - actual)
        mae = np.mean(np.abs(forecast - actual))
        mpe = np.mean((forecast - actual)/actual)
        rmse = np.mean((forecast - actual)**2)**.5
        corr = np.corrcoef(forecast, actual)[0,1]
        mins = np.amin(np.hstack([forecast[:,None], actual[:,None]]), axis=1)
        maxs = np.amax(np.hstack([forecast[:,None], actual[:,None]]), axis=1)
        minmax = 1 - np.mean(mins/maxs)
        return (pd.Series({'mape':mape, 'me':me, 'mae': mae,
                'mpe': mpe, 'rmse':rmse, 'minmax':minmax, 'corr': corr}))

    def plot_test_forecast(self, test_series):
        if not self.auto:
            ax = self.df.plot()
            test_series.plot(ax = ax, color = 'g')
            fc = self.model_fit.forecast(len(test_series))
            fc.plot(ax = ax, color = 'r')
            plt.show()
        else:
            ax = self.df.plot(label = 'Series')
            test_series.plot(ax = ax, color = 'g')
            fc = self.model_fit.predict(len(test_series))
            fc.plot(ax = ax, color = 'r')
            plt.show()
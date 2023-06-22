import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, range_unit_root_test, acf, pacf, ccf, grangercausalitytests, acovf, ccovf
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm

class Trend:
    def __init__(self, df, y: str, x: str):
        self.df = df
        self.x = x
        self.y = y
        self._index_df()

    def _index_df(self):
        idf = pd.DataFrame(self.df[[self.x, self.y]])
        idf[self.x] = pd.to_datetime(idf[self.x], infer_datetime_format = True)
        idf = idf.set_index([self.x])
        self.idf = idf

    def trend(self):
        t = seasonal_decompose(self.idf, model = "multiplicative").trend
        return t

    def plot(self, show_plot = False):
        t = self.trend()
        vals = t.array.dropna().tolist()
        if not show_plot:
            return {
                "title": "Trend of " + self.idf.columns[0] + ":",
                "data": vals,
                "x": [str(i.date())+' '+str(i.time()) for i in self.idf.index.to_list()]
            }
        else:
            t.plot()

    def detrend(self):
        least_squares = OLS(self.df[self.y].values, list(range(self.df.shape[0])))
        result = least_squares.fit()
        fit = pd.Series(result.predict(list(range(self.df.shape[0]))), index = self.df.index)
        detrended = self.df[self.y].values - fit.values

        detrend_df = pd.DataFrame()
        detrend_df['Original'] = self.df[self.y]
        detrend_df['Detrend'] = pd.Series(detrended)
        detrend_df[self.x] = self.df[self.x]
        detrend_df = detrend_df.set_index([self.x])
        return detrend_df


    def plot_detrend(self, show_plot = False, remove_time = False):
        d = self.detrend()
        x_plot =  [str(i.date())+' '+str(i.time()) for i in self.idf.index.to_list()] if not remove_time else [str(i.date()) for i in self.idf.index.to_list()]
        if not show_plot:
            return {
                "title": "Original vs Detrend of " + self.idf.columns[0] + ":",
                "data": d['Original'].values.tolist(),
                "data_detrend": d['Detrend'].values.tolist(),
                "x": x_plot
            }
        else:
            d.plot()

class Seasonality:
    def __init__(self, df, y: str, x: str):
        self.df = df
        self.x = x
        self.y = y

    def _index_df(self):
        idf = pd.DataFrame(self.df[[self.x, self.y]])   
        idf[self.x] = pd.to_datetime(idf[self.x], infer_datetime_format = True)
        idf = idf.set_index([self.x])
        self.idf = idf

    def seasonal(self, model = 'multiplicative'):
        self._index_df()
        t = seasonal_decompose(self.idf, model = model).seasonal
        return t

    def plot(self, show_plot = False, remove_time = False):
        t = self.seasonal()
        x_plot =  [str(i.date())+' '+str(i.time()) for i in self.idf.index.to_list()] if not remove_time else [str(i.date()) for i in self.idf.index.to_list()]
        vals = t.array.dropna().tolist()
        if not show_plot:
            return {
                "title": "Seasonality of " + self.idf.columns[0] + ":",
                "data": vals,
                "x": x_plot
            }
        else:
            t.plot()

    def deseasonalize(self):
        de = self.df[self.y].values / self.seasonal()
        deseasonalize_df = pd.DataFrame()
        deseasonalize_df['Original'] = self.df[self.y]
        deseasonalize_df['Deseasonalized'] = pd.Series(de.values)
        deseasonalize_df[self.x] = self.df[self.x]
        deseasonalize_df = deseasonalize_df.set_index([self.x])
        return deseasonalize_df

    def plot_deseasonalize(self, show_plot = False, remove_time = False):
        d = self.deseasonalize()
        x_plot =  [str(i.date())+' '+str(i.time()) for i in self.idf.index.to_list()] if not remove_time else [str(i.date()) for i in self.idf.index.to_list()]
        if not show_plot:
            return {
                "title": "Original vs Deseasonalized of " + self.idf.columns[0] + ":",
                "data": d['Original'].values.tolist(),
                "data_des": d['Deseasonalized'].values.tolist(),
                "x": x_plot
            }
        else:
            d.plot()

class Stationarity:
    def __init__(self, df, x, y):
        self.df = df
        self.x = x
        self.y = y

    def test_stationarity(self, test = 'adf', show_results = True):
        if test == 'adf':
            dftest = adfuller(self.df[self.y], autolag = 'AIC')
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
            dftest = kpss(self.df[self.y], regression = 'ct')
            print("1. KPSS : ",dftest[0])
            print("2. P-Value : ", dftest[1])   
            print("3. Num Of Lags : ", dftest[2])
            # print("4. Num Of Observations Used For KPSS Regression and Critical Values Calculation :", dftest[3])
            print("4. Critical Values :")
            for key, val in dftest[3].items():
                print("\t",key, ": ", val)
            if dftest[1] > 0.05:
                print("\n\nAs p-value is outside the confidence interval of 95%, series is non-stationary.")
            else:
                print("\n\nAs p-value is inside the confidence interval of 95%, series is stationary.")
        elif test == 'rur':
            dftest = range_unit_root_test(self.df[self.y])
            print("1. RUR Stat : ",dftest[0])
            print("2. P-Value : ", dftest[1])
            # print("3. Num Of Lags : ", dftest[2])
            # print("4. Num Of Observations Used For KPSS Regression and Critical Values Calculation :", dftest[3])
            print("3. Critical Values :")
            for key, val in dftest[2].items():
                print("\t",key, ": ", val)
            if dftest[1] > 0.05:
                    print("\n\nAs p-value is outside the confidence interval of 95%, series is non-stationary.")
            else:
                    print("\n\nAs p-value is inside the confidence interval of 95%, series is stationary.")
        else:
            print("Invlaid Test.")


    def make_stationary(self, method = None, rolling = True, window = 12, test = True ):
        timeseries = self.df[self.y]
        if not method: # only using rolling mean
            rolling_mean = timeseries.rolling(window = window).mean()
            rolling_mean_diff = rolling_mean - rolling_mean.shift()
            transformed = rolling_mean_diff
        else:
            if method == 'log':
                log = pd.Series(np.log(timeseries.values))
                if not rolling: # only log transform
                    log_diff = log - log.shift()
                    transformed = log_diff
                if rolling: # log transform and rolling
                    rolling_log = log.rolling(window = window).mean()
                    diff = rolling_log - rolling_log.shift()
                    transformed = diff
            elif method == 'power':
                power = pd.Series(np.sqrt(timeseries.values))
                if not rolling: # only power transform
                    power_diff = power - power.shift()
                    transformed = power_diff
                if rolling: # power transform and rolling
                    rolling_power = power.rolling(window = window).mean()
                    diff = rolling_power - rolling_power.shift()
                    transformed = diff
            else:
                print("Invalid Method")
                return

        if test:
            p_val = adfuller(transformed.dropna())[1]
            if p_val < 0.05:
                print("p-value: ", p_val, '\nAs p-value is lesser than 0.05, transformed series is stationary.')
            else:
                print("p-value: ", p_val, '\nAs p-value is greater than 0.05, transformed series is non-stationary.')
        self.transformed = transformed
        return transformed.dropna()

    def get_transform_plot_params(self, remove_time = False):
        x_plot =  [str(i.date())+' '+str(i.time()) for i in self.df[self.x].to_list()] if not remove_time else [str(i.date()) for i in self.df[self.x].to_list()]   
        return {
            'title': 'Transformed series',
            'data': self.transformed.dropna().values.tolist(),
            "x": x_plot
        }

class Autocorrelation:
    def __init__(self):
        self.acf = None

    def autocorrelation(self, timeseries, differencing = 1,  n_lags = 20):
        diff = timeseries - timeseries.shift()
        for i in range(differencing-1):
            diff = diff - diff.shift()
        self.diff = diff
        lag_acf = acf(diff.dropna(), nlags = n_lags)
        self.acf = lag_acf

        return {
            'title': 'Autocorrelation plot: differencing = ' + str(differencing),
            'acf_line': lag_acf.tolist(),
            'lower': [-1.96/np.sqrt(len(diff))]*len(lag_acf),
            'upper': [1.96/np.sqrt(len(diff))]*len(lag_acf),
            'zero': [0] * len(lag_acf),
            'x':list(range(len(lag_acf)))
        }

    def durbin_watson_test(self, timeseries, summary = False):
      x = timeseries.index
      y = timeseries.values
      x = sm.add_constant(x)
      reg = sm.OLS(y, x).fit()
      dwtest= durbin_watson(resids = np.array(reg.resid))
      if summary:
        zero, two, four = abs(dwtest), abs(dwtest - 2), abs(dwtest - 4)
        if min(zero, two, four) == zero:
          print("As value of statistic is close to 0, series is positively autocorrelated.")
        elif min(zero, two, four) == two:
          print("As value of statistic is close to 2, series is not autocorrelated.")
        else:
          print("As value of statistic is close to 4, series is negatively autocorrelated.")
      return dwtest

    def estimate_q(self):
        if self.acf is not None:
            u_conf = 1.96/np.sqrt(len(self.diff))
            for i in range(len(self.acf)):
                if self.acf[i] > u_conf:
                    return i, i+1
        else:
            print('Run Autocorrelation.autocorrelation() first')

class PartialAutocorrelation:
    def __init__(self):
        self.pacf = None

    def partial_autocorrelation(self, timeseries, differencing = 1,  n_lags = 20):
        diff = timeseries - timeseries.shift()
        for i in range(differencing-1):
            diff = diff - diff.shift()
        self.diff = diff
        lag_pacf = pacf(diff.dropna(), nlags = n_lags)
        self.pacf = lag_pacf

        return {
            'title': 'Partial autocorrelation plot: differencing = ' + str(differencing),
            'pacf_line': lag_pacf.tolist(),
            'lower': [-1.96/np.sqrt(len(diff))]*len(lag_pacf),
            'upper': [1.96/np.sqrt(len(diff))]*len(lag_pacf),
            'zero': [0] * len(lag_pacf),
            'x':list(range(len(lag_pacf)))
        }

    def estimate_p(self):
        if self.pacf is not None:
            u_conf = 1.96/np.sqrt(len(self.diff))
            for i in range(len(self.pacf)):
                if self.pacf[i] > u_conf:
                    return i,i+1
        else:
            print('Run PartialAutocorrelation.partial_autocorrelation() first.')

class CrossCorrelation:
    def __init__(self, series1, series2, x):
        self.series1 = series1
        self.series2 = series2
        self.time = x

    def cross_correlation(self, lag, summary = False):
        corr = ccf(self.series1.values.tolist(), self.series2.values.tolist(), adjusted = False)
        self.ccf = corr
        if summary:
            print("Max positive correlation is", max(self.ccf) ,"at lag =", np.argmax(self.ccf), "\nMax negative correlation is",min(self.ccf) ," at lag =", np.argmin(self.ccf))
        return corr[lag]

    def plot(self):
        return {
            'legend': [self.series1.name,self.series2.name],
            1: self.series1.values.tolist(),
            2: self.series2.values.tolist(),
            "x": [str(i.date()) for i in self.time.to_list()]
        }

def granger_causality_matrix(data, variables, maxlag = 12, test='ssr_chi2test', verbose=False, summary = False):
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    if summary:
      array = df.values
      for i in range(len(array)):
        for j in range(len(array[i])):
          if array[i][j] < 0.05:
            print(df.columns[j][:-2], " \"granger causes\" ", df.index[i][:-2], " with p-value of ", array[i][j], ".\n", sep = "")
    return df

class Differencing:
  def __init__(self, df, y, x, order = 1):
    self.df = df
    self.y = y
    self.x = x
    self.order = order
    self.diff = self._get_diff().dropna()

  def _diff(self, order):
    return self.df[self.y].diff(order)

  def _get_diff(self):
    return self.df[self.y].diff(self.order)

  def plot_diff(self, remove_time = False):
    x_plot =  [str(i.date())+' '+str(i.time()) for i in self.df[self.x].to_list()] if not remove_time else [str(i.date()) for i in self.df[self.x].to_list()]
    return {
        "data" : self.diff.values.tolist(),
        "x": x_plot
    }

  def adf_test(self):
     p_val = adfuller(self.diff)[1]
     if p_val < 0.05:
         print("p-value: ", p_val, '\nAs p-value is lesser than 0.05, transformed series is stationary.')
     else:
        print("p-value: ", p_val, '\nAs p-value is greater than 0.05, transformed series is non-stationary.')

  def estimate_d(self, max_d = 3):
    for i in range(1, max_d+1):
      p = adfuller(self._diff(i).dropna())[1]
      if p < 0.05:
        return i
    return

class Autocovariance:
  def __init__(self):
    return

  def autocovariance(self, timeseries):
    return acovf(timeseries)

  def plot(self, timeseries, x, remove_time = False):
    x_plot =  [str(i.date())+' '+str(i.time()) for i in x.to_list()] if not remove_time else [str(i.date()) for i in x.to_list()]
    return {
        'title': 'Autocovariance of ' + timeseries.name + ':',
        'y': self.autocovariance(timeseries).tolist(),
        "x": x_plot
    }
    
class CrossCovariance:
  def __init__(self):
    return

  def cross_covariance(self, timeseries1, timeseries2):
    return ccovf(timeseries1, timeseries2)

  def plot(self, timeseries1, timeseries2, x, remove_time = False):
    x_plot =  [str(i.date())+' '+str(i.time()) for i in x.to_list()] if not remove_time else [str(i.date()) for i in x.to_list()]
    return {
        'title': 'Cross covariance of ' + timeseries1.name + ' with ' + timeseries2.name + ':',
        'y': self.cross_covariance(timeseries1, timeseries2).tolist(),
        "x": x_plot
    }

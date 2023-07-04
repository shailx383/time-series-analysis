import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, range_unit_root_test, acf, pacf, ccf, grangercausalitytests, acovf, ccovf
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm

class Trend:
    '''
     class for the trend property
    '''
    def __init__(self, df: pd.DataFrame, y: str, x: str):
        '''
        constructor for Trend class
        
        Args: 
            - df: pandas DataFrame with time series column
            - y: column name of time series data
            - x: column name of datetime axis 
        '''
        self._df = df
        self._x = x
        self._y = y
        self._index_df()

    def _index_df(self):
        '''
        creates an indexed dataframe recognised by decomposer
        '''
        idf = pd.DataFrame(self._df[[self._x, self._y]])
        idf[self._x] = pd.to_datetime(idf[self._x])
        idf = idf.set_index([self._x])
        self._idf = idf

    def _trend(self):
        '''
        returns the trend component of the loaded timeseries
        
        Returns:
            - (pd.Series): trend component of timeseries
        '''
        # print(self._idf)
        t = seasonal_decompose(self._idf[self._idf.columns[0]])
        return t.trend

    def plot(self, remove_time = False):
        '''
        returns dictionary with paramters which can be fed into Apache ECharts
        
        Args:
            - remove_time (bool): True if datetime data must also include time along with date, defaults to False
            
        Returns:
            - (dict):
                - 'title' (str): title of plot
                - 'data' (list): values of trend component
                - 'x' (list): values of x axis, datetime
        '''
        t = self._trend()
        # print(t)
        vals = t.array.dropna().tolist()
        x_plot =  [str(i.date())+' '+str(i.time()) for i in self._idf.index.to_list()] if not remove_time else [str(i.date()) for i in self._idf.index.to_list()]
        return {
            "title": "Trend of " + self._idf.columns[0] + ":",
            "data": vals,
            "x": x_plot
        }

    def detrend(self):
        '''
            returns dataframe with original and de-trended values using OLS linear regression
            
            Args:
                - None
                
            Returns:
                - (pd.DataFrame): dataframe with original and de-trended values using OLS linear regression
        '''
        least_squares = OLS(self._df[self._y].values, list(range(self._df.shape[0])))
        result = least_squares.fit()
        fit = pd.Series(result.predict(list(range(self._df.shape[0]))), index = self._df.index)
        detrended = self._df[self._y].values - fit.values
        detrend_df = pd.DataFrame()
        detrend_df['Original'] = self._df[self._y]
        detrend_df['Detrend'] = pd.Series(detrended).values
        detrend_df[self._x] = self._df[self._x]
        detrend_df = detrend_df.set_index([self._x])
        return detrend_df


    def plot_detrend(self, remove_time = False):
        '''
            returns dictionary with paramters which can be fed into Apache ECharts for detrend plot

            Args:
            - remove_time (bool): True if datetime data must also include time along with date
            
            Returns:
                - (dict):
                    - 'title' (str): title of plot
                    - 'data' (list): values of original series
                    - 'data_detrend (list): values of detrended series
                    - 'x' (list): values of x axis, datetime
        '''
        d = self.detrend()
        x_plot =  [str(i.date())+' '+str(i.time()) for i in self._idf.index.to_list()] if not remove_time else [str(i.date()) for i in self._idf.index.to_list()]
        return {
            "title": "Original vs Detrend of " + self._idf.columns[0] + ":",
            "data": d['Original'].values.tolist(),
            "data_detrend": d['Detrend'].values.tolist(),
            "x": x_plot
        }

class Seasonality:
    '''
        class for the seasonality property of timeseries
    '''
    def __init__(self, df: pd.DataFrame, y: str, x: str):
        '''
            constructor for the Seasonality class
            
            Args:
                - df (pd.DataFrame): dataframe to be analysed
                - y (str): column name of timeseries data to be analysed
                - x (str): column name of datetime axis
        '''
        self._df = df
        self._x = x
        self._y = y

    def _index_df(self):
        '''
            creates an indexed dataframe recognised by decomposer
        '''
        idf = pd.DataFrame(self._df[[self._x, self._y]])   
        idf[self._x] = pd.to_datetime(idf[self._x])
        idf = idf.set_index([self._x])
        self._idf = idf

    def seasonal(self, model = 'multiplicative'):
        '''
            seasonal component of the loaded timeseries
            
            Args:
                - model (str): whether to use additive of multiplicative model for decomposition
                
            Returns:
                - (pd.Series): seasonal component of timeseries
        '''
        self._index_df()
        t = seasonal_decompose(self._idf, model = model).seasonal
        return t

    def plot(self, remove_time = False):
        '''
            plots the seasonal component of the data
            
            Args:
                - remove_time (bool): True if datetime data must also include time along with date
                
        '''
        t = self.seasonal()
        x_plot =  [str(i.date())+' '+str(i.time()) for i in self._idf.index.to_list()] if not remove_time else [str(i.date()) for i in self._idf.index.to_list()]
        vals = t.array.dropna().tolist()
        return {
            "title": "Seasonality of " + self._idf.columns[0] + ":",
            "data": vals,
            "x": x_plot
        }

    def deseasonalize(self):
        de = self._df[self._y].values / self.seasonal()
        deseasonalize_df = pd.DataFrame()
        deseasonalize_df['Original'] = self._df[self._y]
        deseasonalize_df['Deseasonalized'] = pd.Series(de.values).values
        deseasonalize_df[self._x] = self._df[self._x]
        deseasonalize_df = deseasonalize_df.set_index([self._x])
        return deseasonalize_df

    def plot_deseasonalize(self, show_plot = False, remove_time = False):
        d = self.deseasonalize()
        x_plot =  [str(i.date())+' '+str(i.time()) for i in self._idf.index.to_list()] if not remove_time else [str(i.date()) for i in self._idf.index.to_list()]
        if not show_plot:
            return {
                "title": "Original vs Deseasonalized of " + self._idf.columns[0] + ":",
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

    def autocorrelation(self, timeseries, differencing = 0,  n_lags = 20):
        
        df = timeseries
        
        for i in range(differencing):
            df = df.diff().dropna()
            
        acf_vals, confint = acf(df, alpha = 0.05)
        return {
            'title': f'Autocorrelation plot of {timeseries.name}, order = {differencing}',
            'y': acf_vals.tolist(),
            'x': list(range(len(acf_vals))),
            'upper': (confint[:, 1] - acf_vals).tolist(),
            'lower': (confint[:, 0] - acf_vals).tolist()
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
  
class PartialAutocorrelation:
    def __init__(self):
        self.pacf = None

    def partial_autocorrelation(self, timeseries, differencing = 0,  n_lags = 20):
        df = timeseries
        
        for i in range(differencing):
            df = df.diff().dropna()
            
        pacf_vals, confint = pacf(df, alpha = 0.05)
        return {
            'title': f'Partial Autocorrelation plot of {timeseries.name}, order = {differencing}',
            'y': pacf_vals.tolist(),
            'x': list(range(len(pacf_vals))),
            'upper': (confint[:, 1] - pacf_vals).tolist(),
            'lower': (confint[:, 0] - pacf_vals).tolist()
        }

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

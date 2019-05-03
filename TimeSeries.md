<p align="center"> 
<img src="https://github.com/emunozlorenzo/MasterDataScience/blob/master/img/image2.png">
</p>

# TIME SERIES WITH PYTHON
___

![alt text](https://github.com/emunozlorenzo/MasterDataScience/blob/master/img/icon2.png "Logo Title Text 1") [Eduardo Muñoz](https://www.linkedin.com/in/eduardo-mu%C3%B1oz-lorenzo-14144a144/)

## Training and Testing Data
```python
train_data = df.iloc[:109] # .loc[:'1940-01-01']
test_data = df.iloc[108:]
```
## Error Trend Seasonality Decomposition

- You have to watch the range of seasonal plot and check if this range affects your main data in order to know if your time series has seasonal parameters
- For instance: if your seasonal range is between 1000 and -1000 and your main data are around 1E6, then your model don't have seasonal component (ARIMA)

```python
# model = 'additive' 'multiplicative'
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df1['Inventories'],model='add')
from pylab import rcParams
rcParams['figure.figsize'] = 12,6
result.plot();
```
## General Forecasting Models
___
1. Choose a model
2. Split Data into Train and Test Sets
3. Fit the Model on Training Set
4. Evaluate Model on Test Set
5. Refit Model on Entire Dataset
6. Forecast for Future Data

## 1. Reading our Dataset

```python
df = pd.read_csv('./airline_passengers.csv',index_col='Month',parse_dates=True)
df.index.freq= 'MS'
```
## 2. Stationary Data
### How Does a Stationary Series look like? 

- Constant Average
- Constant Variance
- Autocovariance does not depend on the time

_Data don't show any trend and seasonality_
### How Does a Non Stationary Series look like? 

- It shows trend
- It shows seasonality

### Augmented Dickey Fuller Test

```python
from statsmodels.tsa.stattools import adfuller
dftest = adfuller(df['Thousand of Passengers'], autolag='AIC')
print('Test statistic: {}'.format(dftest[0]))
print('p-value: {}'.format(dftest[1]))
print('Lag: {}'.format(dftest[2]))
print('Number of observations: {}'.format(dftest[3]))
for key, value in dftest[4].items():    
    print('Critical Value ({}) = {}'.format(key, value))
```

```python
from statsmodels.tsa.stattools import adfuller

def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")
```

### Differencing

```python
# Differencing once
from statsmodels.tsa.statespace.tools import diff
diff(df['col'],k_diff=1) #.plot(figsize=(12,6),title='Time Series Differenced');
```
## 3. ACF PACF 
```python
from statsmodels.tsa.stattools import acf, pacf, pacf_yw
# ACF
acf(df['col']) # Autocorrelation Function
# PACF
pacf(df['a'],nlags=4, method='ywunbiased') # ywmle and ols
pacf_yw(df['col'],nlags=4,method='mle') # maximum likelihood estimation
pacf_yw(df['col'],nlags=4,method='unbiased') # the statsmodels default
pacf_ols(df['col'],nlags=4) # This provides partial autocorrelations with ordinary least squares (OLS) estimates for each lag instead of Yule-Walker
```
### Plotting lag_plot
```python
# Non Stationary Data
from pandas.plotting import lag_plot
lag_plot(df1['Thousands of Passengers'],lag=1)
```
<p align="center"> 
<img src="https://github.com/emunozlorenzo/MyCheatSheets/blob/master/img/lag_plot_NonStationary.png">
</p>

```python
# Stationary Data
from pandas.plotting import lag_plot
lag_plot(df2['Births'],lag=1)
```

<p align="center"> 
<img src="https://github.com/emunozlorenzo/MyCheatSheets/blob/master/img/lag_plot_Stationary.png">
</p>

### Plotting ACF PACF
#### Non Stationary Data ACF

```python
from statsmodels.graphics.tsaplots import plot_acf
# just 40 lags is enough
plot_acf(df1['Thousands of Passengers'],lags=40,title='Autocorrelation Non Stationary Data')
```
___Shaded region is a 95 percent confidence interval___

___Correlation values OUTSIDE of this confidence interval are VERY HIGHLY LIKELY to be a CORRELATION___

<p align="center"> 
<img src="https://github.com/emunozlorenzo/MyCheatSheets/blob/master/img/ACF_NonStationary.png">
</p>

#### Stationary Data ACF PACF

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# just 40 lags is enough
# shaded region is a 95 percent confidence interval
# Correlation values OUTSIDE of this confidence interval are VERY HIGHLY LIKELY to be a CORRELATION
plot_acf(df2['Births'],lags=40,title='Autocorrelation Stationary Data')
```

<p align="center"> 
<img src="https://github.com/emunozlorenzo/MyCheatSheets/blob/master/img/ACF_Stationary.png">
</p>

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# just 40 lags is enough
# shaded region is a 95 percent confidence interval
# Correlation values OUTSIDE of this confidence interval are VERY HIGHLY LIKELY to be a CORRELATION
plot_pacf(df2['Births'],lags=40,title='Autocorrelation Stationary Data PACF')
```

___PACF works better with Stationary Data___

<p align="center"> 
<img src="https://github.com/emunozlorenzo/MyCheatSheets/blob/master/img/PACF_Stationary.png">
</p>

## 4. AR Model
```python
from statsmodels.tsa.ar_model import AR, ARResults
model = AR(train_data['PopEst'])
# Order 1 p=1 AR(1)
AR1fit = model.fit(maxlag=1,method='cmle',trend='c',solver='lbfgs')
# Order 2 p=2 AR(2)
AR2fit = model.fit(maxlag=2,method='cmle',trend='c',solver='lbfgs')
# To know the order and parameter
AR2fit.k_ar # it shows '2'
AR2fit.params # it shows parameters: yt = c + phi1 yt-1 + phi2 yt-2
# Predict
start = len(train_data)
end=len(train_data) + len(test_data) - 1
pred1 = AR1fit.predict(start=start,end=end)
pred2 = AR2fit.predict(start=start,end=end)
# Plot
test_data.plot(figsize=(12,6))
pred1.plot(legend=True)
pred2.plot(legend=True);
```
### Statsmodel can choose the right order for us

```python
from statsmodels.tsa.ar_model import AR, ARResults
# Instantiate
model = AR(train_data['PopEst'])
# Fit
ARfit = model.fit(ic='t-stat')
ARfit.k_ar # to know the right order
ARfit.params # to know all the parameters
# Predict
start = len(train_data)
end=len(train_data) + len(test_data) - 1
pred = ARfit.predict(start=start,end=end)
# Metrics
from sklearn.metrics import mean_squared_error
mean_squared_error(test_data['PopEst'),pred)
# Plot
test_data.plot(figsize=(12,6))
pred.plot(legend=True)
# Forecast for Furture Values
model = AR(df['PopEst']) # Refit on the entire Dataset
ARfit = model.fit() # Refit on the entire Dataset
forecasted_values = ARfit.predict(start=len(df),end=len(df)+12) # Forecasting 1 year = 12 months
# Plotting
df['PopEst'].plot(figsize=(12,6))
forecasted_values.plot(legend=True);
```

<p align="center"> 
<img src="https://github.com/emunozlorenzo/MyCheatSheets/blob/master/img/AR_img.png">
</p>

<p align="center"> 
<img src="https://github.com/emunozlorenzo/MyCheatSheets/blob/master/img/AR_Forecast_img.png">
</p>

## 4. Metrics
```python
from statsmodels.tools.eval_measures import mse, rmse, meanabs
# Alternative:
# from sklearn.metrics import mean_squared_error
MSE = mse(test_data,preds)
RMSE = rmse(test_data,preds)
MAE = meanabs(test_data,preds)
from statsmodels.tools.eval_measures import aic, bic
```
## 5. Pyramid ARIMA
- The most effective way to get good fitting models
- pmdarima uses AIC as a metric to compare various ARIMA models

```python
# intallation
pip3 install pmdarima
```

```python
# Stationary Dataset
from pmdarima import auto_arima
# Models
stepwise_fit = auto_arima(df2['Births'],start_p=0,start_q=0,max_p=6,max_q=3,seasonal=False,trace=True)
# Best Model
stepwise_fit.summary()
```

```python
# Non Stationary Dataset
from pmdarima import auto_arima
# In this case the dataset has seasonality and m is monthly = 12
stepwise_fit = auto_arima(df1['Thousands of Passengers'],start_p=0,start_q=0,max_p=6,max_q=4,seasonal=True,trace=True,,m=12)
print(stepwise_fit)
# Best Model
stepwise_fit.summary()
```

## 6. ARMA

```python
# 1. Augmented Dickey-Fuller Test to check this Time Series is Stationary ###
from statsmodels.tsa.stattools import adfuller # def adf_test(series,title=''): # See Dickey-Fuller Func
adf_test(df1['Births'], 'Dickey-Fuller Test Births')
# 2. Pyramid Arima to know the right orders p q automatically 
from pmdarima import auto_arima
auto_arima(df1['Births'],seasonal=False,trace=True).summary()
# 3. Train Test Split 
# If We want to forecast one month (30 days), then our Testing Dataset has to be => 30 days
train = df1.iloc[:90]
test = df1.iloc[90:]
# 4. ARMA Model
from statsmodels.tsa.arima_model import ARMA, ARMAResults
model = ARMA(train['Births'],order=(2,2)) # Instantiate
results = model.fit() # Fit
results.summary()
# 5. Predictions
start = len(train)
end = len(train) + len(test) - 1
preds = results.predict(start=start,end=end).rename('ARMA (p,q) Predictions')
# 6. Plotting
test['Births'].plot(figsize=(12,6))
preds.plot(legend=True);
```

## 7. ARIMA

```python
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf # Get parameters watching theses plots
from pmdarima import auto_arima # Get parameters automatically
```






### To avoid warnings

```python
# To avoid seeing warnings
import warnings
warnings.filterwarnings('ignore')
```

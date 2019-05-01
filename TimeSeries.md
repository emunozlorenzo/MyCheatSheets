# TIME SERIES WITH PYTHON
___

## Training and Testing Data
```python
train_data = df.iloc[:109] # .loc[:'1940-01-01']
test_data = df.iloc[108:]
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


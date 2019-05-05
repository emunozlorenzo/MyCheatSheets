<p align="center"> 
<img src="https://github.com/emunozlorenzo/MasterDataScience/blob/master/img/image2.png">
</p>

# TIME SERIES: Facebook's Prophet Library
___

![alt text](https://github.com/emunozlorenzo/MasterDataScience/blob/master/img/icon2.png "Logo Title Text 1") [Eduardo Muñoz](https://www.linkedin.com/in/eduardo-mu%C3%B1oz-lorenzo-14144a144/)

## Documentation

[Forecasting at Scale by Sean Taylor and Benjamin Letham](https://peerj.com/preprints/3190.pdf)

## Installation

https://facebook.github.io/prophet/docs/installation.html#python

```sh
pip3 install fbprophet
```

## 1. Reading our Dataset

- The input to Prophet is always a dataframe with two columns: **ds** and **y**. 
- The **ds** (datestamp) column should be of a **format expected by Pandas**, ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp. 
- The **y** column must be **numeric**, and represents the measurement we wish to forecast.

```python
df = pd.read_csv('../UPDATE-TSA-NOTEBOOKS/Data/BeerWineLiquor.csv')
```
|    | date     | beer   |
|:---|:---------|:-------|
| 0  | 1/1/1992 | 1509   |
| 1  | 2/1/1992 | 1541   |
| 2  | 3/1/1992 | 1597   |
| 3  | 4/1/1992 | 1675   |
| 4  | 5/1/1992 | 1822   |

```python
# Change the names
df.columns = ['ds','y']
# Make sure ds column is a pandas datetime object
df['ds'] = pd.to_datetime(df['ds'])
```

|    | ds                  | y    |
|:---|:--------------------|:-----|
| 0  | 1992-01-01 00:00:00 | 1509 |
| 1  | 1992-02-01 00:00:00 | 1541 |
| 2  | 1992-03-01 00:00:00 | 1597 |
| 3  | 1992-04-01 00:00:00 | 1675 |
| 4  | 1992-05-01 00:00:00 | 1822 |

- By default Prophet is going to expect daily data

## 2. Train Test Split

```python
print(len(df)) # to know how many rows we have
train = df.iloc[:576]
test = df.iloc[576:]
```

## 3. Model

```python
from fbprophet import Prophet
m = Prophet()
m.fit(train)
```

## 4. Predictions

```python
# In Stead of doing df.index.freq = 'MS' as we do in Statsmodel
future = m.make_future_dataframe(periods=12,freq='MS')
preds = m.predict(future)
preds.head() # to find out more about the output
# yhat, yhat_lower and yhat_upper are probably the most important terms of this DF
preds[['ds','yhat_lower','yhat_upper','yhat']].tail(24)
```

## 4. Plotting the Results

```python
ax = preds.plot(x='ds',y='yhat',label='Predictions',legend=True,figsize=(12,6))
test.plot(x='ds',y='y',label='Test Data',legend=True,ax=ax,xlim=('2018-01-01','2019-01-01'));
```

```python
m.plot(preds);
```

<p align="center"> 
<img src="https://github.com/emunozlorenzo/MyCheatSheets/blob/master/img/prophet_output.png">
</p>

```python
# Trend and Seasonality
m.plot_components(preds);
```
<p align="center"> 
<img src="https://github.com/emunozlorenzo/MyCheatSheets/blob/master/img/prophet_output2.png">
</p>


## 5. Evaluate the Model

### 5.1 RMSE

```python
from statsmodels.tools.eval_measures import rmse
# Alternative:
# from sklearn.metrics import mean_squared_error
RMSE = rmse(test['y'],preds['yhat'][-12:])
RMSE # it should be interesting to compare this value with test['y'].mean()
```
### 5.2 Prophet's Metrics

```python
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import plot_cross_validation_metric

# The initial period should be long enough to capture all of the components of the model, 
# in particular seasonalities and extra regressors: at least a year for yearly seasonality,
# at least a week for weekly seasonality, etc.

# Initial training period
initial = 5 * 365
initial = str(initial) + ' days'
# Period lenght that we are going to perform the cross validation. 
# How many times to fold?
period = 5 * 365
period = str(period) + ' days'
# Horizon of prediction for essentially each fold. 
# How far out do you want to forecast for each period?
horizon = 365
horizon = str(horizon) + ' days'

df_cv = cross_validation(m,initial=initial,period=period,horizon=horizon)
performance_metrics(df_cv)

plot_cross_validation_metric(df_cv,metric='rmse');
```

## 6. Forecast

```python
from fbprophet import Prophet
m = Prophet()
m.fit(df)
# In Stead of doing df.index.freq = 'MS' 
future = m.make_future_dataframe(periods=12,freq='MS')

forecast = m.predict(future)
m.plot(forecast);
m.plot_components(forecast);
```

## Trend Line

```python
# 1. Loading Libraries
import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet
# 2. Reading the Dataset
df = pd.read_csv('../UPDATE-TSA-NOTEBOOKS/Data/HospitalityEmployees.csv')
df.columns = ['ds','y']
df['ds'] = pd.to_datetime(df['ds'])
# 3. Fitting the Model
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=12,freq='MS')
# 4. Predcitions or Forecasting
forecast = m.predict(future)
# 5. Main Changes in Trend Line
# It shows the major points where the trend line happened to change
from fbprophet.plot import add_changepoints_to_plot
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(),m,forecast);
```

<p align="center"> 
<img src="https://github.com/emunozlorenzo/MyCheatSheets/blob/master/img/prophet_output3.png">
</p>


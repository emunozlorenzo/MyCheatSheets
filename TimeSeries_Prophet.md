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

## 2. Model

```python
from fbprophet import Prophet
m = Prophet()
m.fit(df)
```

## 3. Forecast

```python
# In Stead of doing df.index.freq = 'MS' as we do in Statsmodel
future = m.make_future_dataframe(periods=24,freq='MS')
forecast = m.predict(future)
forecast.head() # to find out more about the output
# yhat, yhat_lower and yhat_upper are probably the most important terms of this DF
forecast[['ds','yhat_lower','yhat_upper','yhat']].tail(24)
```

## 4. Plotting the Results

```python
m.plot(forecast);
```
<p align="center"> 
<img src="https://github.com/emunozlorenzo/MyCheatSheets/blob/master/img/prophet_output.png">
</p>

```python
# Trend and Seasonality
m.plot_components(forecast);
```
<p align="center"> 
<img src="https://github.com/emunozlorenzo/MyCheatSheets/blob/master/img/prophet_output2.png">
</p>

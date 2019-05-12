<p align="center"> 
<img src="https://github.com/emunozlorenzo/MasterDataScience/blob/master/img/image2.png">
</p>

# TIME SERIES WITH KERAS
___

![alt text](https://github.com/emunozlorenzo/MasterDataScience/blob/master/img/icon2.png "Logo Title Text 1") [Eduardo Muñoz](https://www.linkedin.com/in/eduardo-mu%C3%B1oz-lorenzo-14144a144/)

## Installation

```python
pip3 install tensorflow
pip3 install keras
```

## Deep Learning using KERAS

1.  Define the **Sequential** Model Object
3.  Add **layers** to it
4.  Fit the Model on Training Set for a chosen number of **epochs**

**Epoch**: One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE

**Batch**: Since one epoch is too big to feed to the computer at once we divide it in several smaller batches.

#### Why we use more than one Epoch?

I know it doesn’t make sense in the starting that — passing the entire dataset through a neural network is not enough. And we need to pass the full dataset multiple times to the same neural network. But keep in mind that we are using a limited dataset and to optimise the learning and the graph we are using **Gradient Descent** which is an **_iterative_** process. So, _updating the weights with single pass or one epoch is not enough._

> One epoch leads to underfitting of the curve in the graph (below).

![](https://cdn-images-1.medium.com/max/800/1*i_lp_hUFyUD_Sq4pLer28g.png)

As the number of epochs increases, more number of times the weight are changed in the neural network and the curve goes from **underfitting** to **optimal** to **overfitting** curve.

```
We can divide the dataset of 2000 examples into batches of 500 then it will take 4 iterations to complete 1 epoch.
```
To find out more visit [this article](https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9)

## 1. Reading our Dataset

```python
df = pd.read_csv('../UPDATE-TSA-NOTEBOOKS/Data/Alcohol_Sales.csv', index_col='DATE',parse_dates=True)
df.index.freq = 'MS'
```
## 2. Train Test Split

```python
print(len(df)) # to know how many rows we have
train = df.iloc[:310]
test = df.iloc[310:]
```

## 3. Scale your Data

Normalization is a rescaling of the data from the original range so that all values are within the range of 0 and 1.

Normalization can be useful, and even required in some machine learning algorithms when your time series data has input values with differing scales.It may be required for algorithms, like k-Nearest neighbors, which uses distance calculations and Linear Regression and **Artificial Neural Networks** that weight input values.

```python
from sklearn.preprocessing import MinMaxScaler

# Instantiate
scaler = MinMaxScaler()

# Fit: Find the max value in the training dataset
scaler.fit(train)
print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))

# normalize the dataset
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

# inverse transform
inversed_scaled_train = scaler.inverse_transform(scaled_train)
inversed_scaled_test = scaler.inverse_transform(scaled_test)
```


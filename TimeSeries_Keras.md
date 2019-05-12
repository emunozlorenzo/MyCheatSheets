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

## 4. Preprocessing your Data to work with Keras and TimeSeries

Take a sequence of data-points gathered at equal intervals  to produce batches for training/validation.

```python
from keras.preprocessing.sequence import TimeseriesGenerator
# len(train_generator) = len(scaled_train) - n_inputs
n_input = 12      # seasonality every 12 months
n_features = 1    # How many columns you have...for TS it should be just one
# TimeSeries Generator
train_generator = TimeseriesGenerator(scaled_train,scaled_train,length=n_input,batch_size=1)
```

## 5. Model

```python
from keras.models import Sequential
from keras.layers import Dense, LSTM
# units = number of neurons, activation = activation function, input_shape, input_dim is the input dimesion
model = Sequential()
# First Layer LSTM
model.add(LSTM(units=150,input_shape=(n_input,n_features),activation='relu'))
# Output Layer
model.add(Dense(units=1))
# Compile
model.compile(loss='mse',optimizer='adam')
# Summary
model.summary()
```
```python
# Fitting our Neuronal Network
# epochs really depends on how large your dataset is...and how much training data you have....
model.fit_generator(train_generator,epochs=25)
```

## 6. Plotting Loss vs Epochs

- Two ways

```python
# Plotting loss vs epochs
loss = model.history.history['loss']
epochs = range(len(loss))
plt.plot(epochs,loss);
```
```python
h = model.fit_generator(train_generator,epochs=25)

def plot_metric(history, metric):
    history_dict = history.history
    values = history_dict[metric]
    if 'val_' + metric in history_dict.keys():  
        val_values = history_dict['val_' + metric]

    epochs = range(1, len(values) + 1)

    if 'val_' + metric in history_dict.keys():  
        plt.plot(epochs, val_values, label='Validation')
    plt.semilogy(epochs, values, label='Training')

    if 'val_' + metric in history_dict.keys():  
        plt.title('Training and validation %s' % metric)
    else:
        plt.title('Training %s' % metric)
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.legend()

    plt.show()  

plot_metric(h,'loss')
```

## 7. Predict

```python
# holding my predictions
test_predictions = []

# last n_input points from our training dataset
first_eval_batch = scaled_train[-n_input:]
# reshape this to the right format to work with RNN (same format as TTimeseriesGenerator)
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test)):
    
    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0]
    
    # store prediction
    test_predictions.append(current_pred) 
    
    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
```
## 8. Plotting Predictions
```python
# Inverse Transformation
true_predictions = scaler.inverse_transform(test_predictions)
test['Predictions'] = true_predictions
test.plot(figsize=(12,6));
```

## 9. Evaluate the Model

```python
from statsmodels.tools.eval_measures import rmse
error = rmse(test['Sales'],test['Predictions']) # Compare it with test.mean()
error
```

## 10. Save your Model

```python
model.save('mymodel.h5')
```

## 11. Load the Model

```python
from keras.models import load_model
new_model = load_model('mymodel.h5')
# Summary
new_model.summary()
# etc..
```


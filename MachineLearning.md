# MACHINE LEARNING CHEAT SHEET
___

![alt text](https://github.com/emunozlorenzo/MasterDataScience/blob/master/img/icon2.png "Logo Title Text 1") [Eduardo Muñoz](https://www.linkedin.com/in/eduardo-mu%C3%B1oz-lorenzo-14144a144/)

# Supervised Learning
### Training and Testing Data
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state = 4) 
```

# 1. Regression
___

## Linear Regression Statsmodel

```python
import statsmodels.formula.api as smf
lm = smf.ols(formula='Sales~TV+Radio', data=training).fit
lm.pvalues
lm.summary()
sales_pred = lm.predict(testing)
```
## Linear Regression Sklearn
y=β0+β1x1+β2x2+...+βnxn
```python
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train,y_train)
linreg.intercept_ # β0
linreg.coef_ # β1 β2 … βn
y_pred = linreg.predict(X_test)
```
### KNeighbors Regressor
```python
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=30)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
```
### Decision Tree Regressor

Main parameters
* Max_depth: Number of Splits
* Min_samples_leaf: Minimum number of observations per leaf

*min_samples_split 2 by default!!! Be careful: overfitting!!!*

```python
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(max_depth=5,min_samples_leaf=20,min_samples_split=20)
tree.fit(X_train,y_train)
y_pred = tree.predict(X_test)
```
### Gradient Boosted Trees Regressor

```python
from sklearn.ensemble import GradientBoostingRegressor
gbm = GradientBoostingRegressor(max_depth=4,n_estimators=100, learning_rate=0.1)
gbm.fit(X_train,y_train)
y_pred = gbm.predict(X_test)
```
# 2. Classification
___
### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg # print this you’ll see all the default values
logreg.fit(X_train,y_train)  # Training our model
y_pred = logreg.predict(X_test) # ex:arr([2]) predict class 2
y_pred_prob = logreg.predict_proba(X_test) # ex:arr([[0.01,0.19,0.8]])predict class 2
```
### KNeighbors Classifier

```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn # print this you’ll see all the default values
knn.fit(X_train,y_train) # Training our model
y_pred=knn.predict(X_test) # ex:arr([2]) predict class 2
y_pred_prob = knn.predict_proba(X_test) # ex:arr([[0,0,1]])predict class 2
```

### Support Vector Machine SVM

Parameters:

* C: Sum of Error Margins
* kernel:
        - linear: line of separation
        - rbf: circle of separation
            * Additional param gamma: Inverse of the radius
        - poly: curved line of separation
            * Additional param degree: Degree of the polynome

```python
from sklearn.svm import SVC
svm = SVC(kernel='linear',C=10)
svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)
y_pred_prob = svm.predict_proba(X_test)
```
### Decision Tree Classifier

```python
# Gini by default, min_samples_split 2 by default!!! Be careful: overfitting!!!
from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier(criterion=’entropy’,min_samples_split=20,min_samples_leaf=20, random_state=99)
tree.fit(X_train,y_train)
y_pred = tree.predict(X_test)
```
### Multinomial Naive Bayes 
```python
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train,y_train)
y_pred = nb.predict(X_test_dtm)
y_pred_prob = nb.predict_proba(X_test)
```

# 3. Metrics
___
## 3.1 Regression Metrics

- MAE is the easiest to understand, because it's the average error.
- MSE is more popular than MAE, because MSE "punishes" larger errors.
- RMSE is more popular than MSE, RMSE is interpretable in the "y" units.

### Mean Absolute Error MAE
```python
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,y_pred)
```
### Mean Absolute Percentage Error MAPE
```python
np.mean(np.abs(y_test-y_pred)/y_test)
```

### Mean Squared Error MSE
```python
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,y_pred)
```
### Root Mean Squared Error RMSE
```python
import numpy as np
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(y_test,y_pred))
```
### R2 Score
```python
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
```
### Correlation
```python
# Direct Calculation
np.corrcoef(reg.predict(X_test),y_test)[0][1]
# Custom Scorer
from sklearn.metrics import make_scorer
def corr(y_pred,y_test):
    return np.corrcoef(y_pred,y_test)[0][1]
# Put the scorer in cross_val_score
cross_val_score(reg,X,y,cv=5,scoring=make_scorer(corr))
```

### Bias
```python
# Direct Calculation
np.mean(reg.predict(X_test)-y_test)
# Custom Scorer
from sklearn.metrics import make_scorer
def bias(y_pred,y_test):
    return np.mean(y_pred-y_test)
# Put the scorer in cross_val_score
cross_val_score(reg,X,y,cv=5,scoring=make_scorer(bias))
```


## 3.2 Classification Metrics
### Accuracy Score 

- percentage of correct predictions

```python
from sklearn import metrics 
metrics.accuracy_score(y_test, y_pred) # metric score
```
```python
knn.score(X_test,y_test) # estimator score method
# Cross Validation
cross_val_score(knn,X,y,scoring="accuracy")
```
### Null Accuracy 

- predicting the most frequent class 

```python
max(y_test.mean(), 1 - y_test.mean()) # coded as 0/1 
y_test.value_counts().head(1) / len(y_test) # multiclass
```

### Classification Report 
- Precision, Recall, F1 and Support
```python
print(metrics.classification_report(y_test,y_pred))
```
### Confusion Matrix
```python
from sklearn import metrics
metrics.confusion_matrix(y_test,y_pred) 
```
#### Plot a Confusion Matrix with Seaborn
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
ax = sns.heatmap(confusion_matrix(y_test,y_pred) , annot=True, fmt="d",cmap="YlGnBu")
ax.set(xlabel='Predicted Values', ylabel='Actual Values',title='Confusion Matrix');
```

### Sensitivity (True Positive Rate or Recall)
```python
from sklearn import metrics
metrics.recall_score(y_test, y_pred)
# Cross Validation
cross_val_score(clf,X,y,scoring="recall")
```
### Specificity (True Negative Rate in 0/1 code)

- metrics.classification_report recall for 0

```python
print(metrics.classification_report(y_test,y_pred))
```
### Precision 
- precision predicting positive instances
```python
from sklearn import metrics
metrics.precision_score(y_test, y_pred)
# Cross Validation
cross_val_score(clf,X,y,scoring="precision")
```

### ROC curve

*how can i see sensitivity and specificity being affected by various thresholds, without actually changing the threshold? Best Result: +++ Sensi(TP) and +++ Speci(TN)
but wait!!! Sensi and speci have an inverse relationship!!!*
```python
from sklearn import metrics 
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob[:,1])
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC Curve')
plt.xlabel('False Positive Rate (1 - Specificity)\n+++Specificity---')
plt.ylabel('True Positive Rate (Sensitivity)\n---Sensitivity+++')
plt.grid(True)
```
```python
# Another Way
from sklearn.metrics import roc_curve
# We chose the target
target_pos = 1 # Or 0 for the other class
fp,tp,_ = roc_curve(y_test,y_pred_prob[:,target_pos])
plt.plot(fp,tp)
```
### AUC

*AUC is useful as a single number summary of classifier performance.
If you randomly chose one positive and one negative observation, AUC represents the likelihood that your classifier will assign a higher predicted probability to the positive observation.
AUC is useful even when there is high class imbalance (unlike classification accuracy).*

```python
y_pred_prob = logreg.predict_proba(X_test)
from sklearn import metrics 
metrics.roc_auc_score(y_test, y_pred_prob[:,1])
# Cross-Validation AUC
from sklearn.model_selection import cross_val_score
cross_val_score(logreg, X, y, cv=10, scoring='roc_auc').mean()
```
```python
# Metrics
from sklearn.metrics import roc_curve, auc
fp,tp,_ = roc_curve(y_test,y_pred_prob[:,1])
auc(fp,tp)
# Cross Validation
cross_val_score(clf,X,y,scoring="roc_auc")
```
### CHANGING THRESHOLD (CLASSIFICATION MODEL)
```python
from sklearn.preprocessing import binarize
y_pred = algori.predict(X_test)
y_pred_prob = algori.predict_poba(X_test)
y_newpred = binarize([y_pred_prob[:, 1]], threshold=0.3)[0]
metrics.confusion_matrix(y_test,y_newpred)
```
# 4. Cross Validation 
___

## Cross Validation Regression
```python
# Load the library
from sklearn.model_selection import cross_val_score
# We calculate the metric for several subsets (determine by cv)
# With cv=5, we will have 5 results from 5 training/test
cross_val_score(reg,X,y,cv=5,scoring="neg_mean_squared_error")
```
```python
# Another example
from sklearn.model_selection import cross_val_score
linreg = LinearRegression()
scores = cross_val_score(linreg, X, y, cv=10, scoring=’accuracy’)
scores.mean()
```
```python
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
from sklearn.model_selection import cross_val_score
scores = cross_val_score(linreg, X, y, cv=10, scoring='neg_mean_squared_error')
mse_scores = -scores
rmse_scores= np.sqrt(mse_scores)
rmse_scores.mean()
```
## Cross Validation Classification

```python
from sklearn.model_selection import cross_val_score
knn=KNeighborsClassifier(n_neighbors=5)
cross_val_score(knn, X, y, cv=5, scoring='accuracy').mean()
```
#### Find the optimal value of K for KNN
```python
k_range = range(1,31)
k_scores = []
for k in k_range:
   	knn=KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10,scoring='accuracy')
   	k_scores.append(scores.mean())
    print('K = %s, Score_mean = %s'%(k,scores.mean()))
plt.plot(k_scores)
```
- Low values of K = Low Bias Model and High Variance
- High values of K = High Bias and Low Variance
- K = 1 High Complexity Model, High Variance and Low Bias
- K >>> Low Complexity Model, High Bias and Low Variance

# 5. GridSearch CV
___
```python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

grid = GridSearchCV(KNeighborsRegressor(),
                    param_grid={"n_neighbors":np.arange(1,50),
                                "weights":['uniform','distance']},
                    scoring='accuracy',
                    n_jobs=-1 # All CPUs working
                    verbose= True)
                    
# Fit will test all of the combinations
grid.fit(X,y)
# Best estimator and best parameters
grid.best_score_
grid.best_estimator_ # Best Trained Model
grid.best_params_

import pandas as pd
pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]

best_model = grid.best_estimator_
best_model.predict(...)
```
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

grid=GridSearchCV(DecisionTreeClassifier(),
                 param_grid={'max_depth':range(1,20),'min_samples_leaf':range(20,100),'min_samples_split':range(20,50)},
                 cv=6,
                 scoring='accuracy',
                 n_jobs=-1,
                 verbose=True)
                 
grid.fit(X,y)
tree_final = grid.best_estimator_
```
# 6. RandomizedSearch CV
___

- Random GridSearchCV: faster and quite accurate

```python
from sklearn.model_selection import RandomizedSearchCV
rand = RandomizedSearchCV(DecisionTreeRegressor(),
                          param_distributions={'max_depth':range(1,10),'min_samples_leaf':range(1,50)},
                          cv=6,
                          scoring='neg_mean_absolute_error',
                          n_iter=10,
                          n_jobs=-1,
                          scoring='accuracy',
                          random_state= 5,
                          verbose= True)
rand.fit(X,y)

rand.best_estimator_
rand.best_params_
rand.best_score_

pd.DataFrame(rand.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
```

# 7. Saving an Delivering a Model
___
```python
import pickle
pickle.dump(knn,open('model.pickle','wb'))
model_loaded = pickle.load(open('model.pickle','rb'))
```

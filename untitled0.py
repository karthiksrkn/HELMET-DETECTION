import sklearn
from sklearn.datasets import load_boston
import pandas as pd
import warnings
from sklearn import linear_model
from sklearn.model_selection import train_test_split
boston=load_boston()  
warnings.simplefilter(action="ignore",category=FutureWarning)
bos=pd.DataFrame(boston.data)
bos.columns = boston.feature_names
print(bos.describe())
bos['PRICE']=boston.target
print(bos)
print(boston.target)
X = bos.drop('PRICE', axis = 1)
Y = bos['PRICE']
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.53, random_state = 3)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, Y_train)
Y_pred= lm.predict(X_test)
plt.scatter(Y_test, Y_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
print(mse)
plt.figure(figsize=(4, 3))
plt.hist(boston.target)
plt.xlabel('price ($1000s)')
plt.ylabel('count')
for index, feature_name in enumerate(boston.feature_names):
    plt.figure(figsize=(4, 3))
    plt.scatter(boston.data[:, index], boston.target)
    plt.ylabel('Price', size=15)
    plt.xlabel(feature_name, size=15)
    plt.tight_layout()
   
import math
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(X_train,Y_train)
train_score=lasso.score(X_train,Y_train)
test_score=lasso.score(X_test,Y_test)
coeff_used = np.sum(lasso.coef_!=0)
print train_score
print test_score
print coeff_used
lasso001 = Lasso(alpha=0.01, max_iter=10e5)
lasso001.fit(X_train,Y_train)
train_score001=lasso001.score(X_train,Y_train)
test_score001=lasso001.score(X_test,Y_test)
coeff_used001 = np.sum(lasso001.coef_!=0)
print train_score001
print test_score001
print coeff_used001
from sklearn.linear_model import Ridge
rr = Ridge(alpha=0.01)
rr.fit(X_train, Y_train)
Ridge_train_score = rr.score(X_train,Y_train)
Ridge_test_score = rr.score(X_test, Y_test)
print Ridge_train_score
print Ridge_test_score
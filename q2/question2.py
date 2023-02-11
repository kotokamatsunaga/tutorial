
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

iris_dataset = load_iris()

X = iris_dataset['data']
y = iris_dataset['target']
y_ones=np.ones_like(y)
y_2d=np.array(y_ones).reshape(len(y),1)
X_new=np.hstack((y_2d,X))

def f(x,b):
    y=1/(1+np.exp(-np.matmul((b.T),x)))
    return y

b=np.reshape(np.random.rand(5),(5,1))
y_pred=f(X_new.T,b) #問題文のX_newが横に結合してるから二度手間?ここで150コのデータについて0-1で予測できた→y_predを3倍して、0-1,1-2,2-3
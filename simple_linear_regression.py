class linearRegression:
  def __init__(self):
    self.m = None
    self.b = None
  def fit(self,x_train,y_train):
    num = 0
    den = 0
    for i in range(x_train.shape[0]):
      num = num + ((x_train[i] - x_train.mean())*(y_train[i] - y_train.mean()))
      den = den + ((x_train[i] - x_train.mean())*(x_train[i] - x_train.mean()))
    self.m = num/den
    self.b = y_train.mean() - (self.m*x_train.mean())
    print(self.m)
    print(self.b)
  def predict(self,x_test):
    print(x_test)
    return self.m * x_test + self.b
 # ///////////////
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
data = pd.read_csv(r"c:\Users\dell\Downloads\score.csv")
data.head()
x = data.iloc[:,0].values
y = data.iloc[:,1].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
lr = linearRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
print(y_pred)
accuracy = r2_score(y_test,y_pred)
print(accuracy)
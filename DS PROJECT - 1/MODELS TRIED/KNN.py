
# KNN MODEL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pylab
%matplotlib inline
import statsmodels.api as smf
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler,RobustScaler
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder

# loading the beverage

beverage = pd.read_excel('C:/Users/HP/Downloads/DATA - Copy.xlsx')
beverage.shape
beverage.info()
beverage=beverage.drop(['Timestamp'], axis = 1)

desc=beverage.describe()
categorical=pd.cut(beverage.allplant,bins=[250,358,462],labels=['low','high'])
beverage.insert(10,'energy_consumption',categorical)


beverage['energy_consumption'].unique()
beverage['energy_consumption'].value_counts()

beverage=beverage.drop(['allplant'], axis = 1)
lb = LabelEncoder()
beverage["energy_consumption"] = lb.fit_transform(beverage["energy_consumption"])
X = np.array(beverage.iloc[:,:]) # Predictors 
Y = np.array(beverage['energy_consumption']) # Target 

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

# MODULE 3 : BUILDING MODEL 
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5000)
knn.fit(X_train, Y_train)

pred = knn.predict(X_test)
pred

#MODULE 4 : MODEL EVALUATION
# Evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, pred))
pd.crosstab(Y_test, pred, rownames = ['Actual'], colnames= ['Predictions']) 


# error on train data
pred_train = knn.predict(X_train)
print(accuracy_score(Y_train, pred_train))
pd.crosstab(Y_train, pred_train, rownames=['Actual'], colnames = ['Predictions']) 


# creating empty list variable 
acc = []

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values

for i in range(3,50,2):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, Y_train)
    train_acc = np.mean(neigh.predict(X_train) == Y_train)
    test_acc = np.mean(neigh.predict(X_test) == Y_test)
    acc.append([train_acc, test_acc])


import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")

# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"bo-")

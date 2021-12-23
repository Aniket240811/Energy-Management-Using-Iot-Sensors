#CLASSIFICATION



# DECISION TREE
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
colnames = list(beverage.columns)


#FINDING CORRELATION COEFFICIENT BETWEEN DIFFERENT FEATURES
correlation = beverage.corr()

   


predictors = colnames[:9]
target = colnames[9]

# MODULE 3 : MODEL BUILDING
# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(beverage, test_size = 0.20,random_state=1)

from sklearn.tree import DecisionTreeClassifier as DT

help(DT)
model = DT(criterion = 'entropy',max_depth=9)
model.fit(train[predictors], train[target])

# MODULE 4 : DATA EVALUATION
# Prediction on Test Data
Y_test_pred = model.predict(test[predictors])
pd.crosstab(test[target], Y_test_pred, rownames=['Actual'], colnames=['Predictions'])

np.mean(Y_test_pred == test[target]) # Test Data Accuracy 

# Prediction on Train Data
Y_train_pred = model.predict(train[predictors])
pd.crosstab(train[target], Y_train_pred, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(Y_train_pred == train[target]) # Train Data Accuracy

from sklearn.metrics import accuracy_score
print(accuracy_score(train[target],Y_train_pred),round(accuracy_score(test[target],Y_test_pred),2))
#PRUNNING:
path=model.cost_complexity_pruning_path(train[predictors],train[target])
alphas=path['ccp_alphas']
alphas
accuracy_train,accuracy_test=[],[]
for i in alphas:
    tree=DT(ccp_alpha=i)
    tree.fit(train[predictors],train[target])    
    Y_train_pred=model.predict(train[predictors])
    Y_test_pred = model.predict(test[predictors])
    
    accuracy_train.append(accuracy_score(train[target],Y_train_pred))
    accuracy_test.append(accuracy_score(test[target],Y_test_pred))

sns.set()
plt.figure(figsize=(14,7))
sns.lineplot(y=accuracy_train,x=alphas,label="Train Accuracy")
sns.lineplot(y=accuracy_test,x=alphas,label="Test Accuracy")
plt.xticks(ticks=np.arange(0.00,2.00,0.50))
plt.show()

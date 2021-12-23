# MODULE 1
### Load all the packages required
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pylab
##%matplotlib inline
import statsmodels.api as smf
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler,RobustScaler
import scipy.stats as stats

# loading the beverage file
beverage = pd.read_excel('Downloads/DATA - Copy1.xlsx')
beverage.head()
beverage.shape
beverage.info()
beverage=beverage.drop(['Timestamp'], axis = 1)
desc=beverage.describe()

# MODULE 2
# Exploratory Data Analysis
## Measures of Central Tendency / First moment business decision
################### 1. Mean #######################
beverage.mean()
#################### 2.Median #################"""
beverage.median()
# Measures of Dispersion / Second moment business decision
################# Variance #################
beverage.var()
################### Stdev ###################"""
beverage.std()
# Third moment business decision
################# Skew ####################
beverage.skew()
#Fourth Moment Business Decision
############## kurtosis #################
beverage.kurt()
# Visualizations
## Univariate analysis
beverage.hist()
### Finding ouliers"""
for i in beverage.iloc[:,:].columns:
  sns.boxplot(beverage[i])
  plt.show()     ##  no outliers ; combined boxplot not showing any outliers
ax=sns.boxplot(data= beverage,orient="h")

# Module 3
### Data Preprocessing
duplicate = beverage.duplicated()
duplicate
sum(duplicate)
### Outlier Treatment"""
# Detection of outliers (find limits for salary based on IQR)
IQR = beverage['RCM'].quantile(0.75) - beverage['RCM'].quantile(0.25)
lower_limit = beverage['RCM'].quantile(0.25) - (IQR * 1.5)
upper_limit = beverage['RCM'].quantile(0.75) + (IQR * 1.5)

outliers_beverage = np.where(beverage['RCM'] > upper_limit, True, np.where(beverage['RCM'] < lower_limit, True, False))
beverage_trimmed = beverage.loc[~(outliers_beverage), ]
beverage.shape, beverage_trimmed.shape

# let's explore outliers in the trimmed dataset
sns.boxplot(beverage_trimmed.RCM);plt.title('Boxplot');plt.show()

############### 2.Replace ###############
# Now let's replace the outliers by the maximum and minimum limit
beverage['RCM'] = pd.DataFrame(np.where(beverage['RCM'] > upper_limit, upper_limit, np.where(beverage['RCM'] < lower_limit, lower_limit, beverage['RCM'])))
sns.boxplot(beverage.RCM);plt.title('Boxplot');plt.show()
# Check for Missing Values
########### check for count of NA'sin each column
beverage.isna().sum()
# Visualizations
# Pair Plot
sns.pairplot(beverage)
# Heat map after Standardization"""
fig, ax = plt.subplots(figsize=(20,20))       
sns.heatmap(beverage.corr(),annot=True, linewidths=.5, ax=ax)
### Model #######
categorical=pd.cut(beverage.allplant,bins=[250,358,462],labels=['low','high'])
beverage.insert(10,'energy_consumption',categorical)
beverage['energy_consumption'].unique()
beverage['energy_consumption'].value_counts()
### Dropping allplant column"""
beverage=beverage.drop(['allplant'], axis = 1)
colnames = list(beverage.columns)
### FINDING CORRELATION COEFFICIENT BETWEEN DIFFERENT FEATURES"""
correlation = beverage.corr()
predictors = colnames[:9]
target = colnames[9]


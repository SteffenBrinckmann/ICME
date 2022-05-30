# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#DOWNLOAD FROM https://github.com/SteffenBrinckmann/ICME
# OPTION 1
# - click on the file -> "RAW"
# - copy-paste into ...
#
# OPTION 2
# - click on green "Code" and download zip-file
dataset= pd.read_csv('ScratchArea.csv') 

#visually check data, first
print(dataset.head())

"""
plt.plot(dataset['Material'], dataset['scratch area 1'],'o')
plt.xlabel('Load in $mN$')
plt.ylabel('Scratch Area in $\mu m^2$')
plt.show()

#sns.displot(dataset['scratch area 1'], bins=20, kde=True)
#plt.show()
sns.countplot(x="Material", hue="Tip", data=dataset)
plt.show()
sns.pairplot(dataset, hue='Material')
plt.show()
"""

sns.heatmap(dataset.corr(), annot=True)
plt.show()


#Arrange columns 
X = dataset.iloc[:, [0,2,5,6] ].values 
y = dataset.iloc[:, 3].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error ,r2_score, mean_absolute_percentage_error
import statsmodels.api as sm

X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                                    test_size = 0.1)
# PREDICTION VERY BAD
regression= LinearRegression()
regression.fit(X_train, y_train)
y_prediction = regression.predict(X_test)

print("R2 value is",r2_score(y_test, y_prediction))
print("mean square error", mean_squared_error(y_test,y_prediction))
print('mean absolute percentage error',mean_absolute_percentage_error(y_test,y_prediction))

#LET ONLY LOOK AT Al
import numpy as np
# Only  thing that changed
subdataset = dataset[np.logical_and(dataset['Material']=='Al',dataset['Tip']==20)]
#--copy pasted start
X = subdataset.iloc[:, [0,2,5,6] ].values
y = subdataset.iloc[:, 3].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0) 
regressor = LinearRegression()
regressor.fit(X_train, y_train) 
y_pred = regressor.predict(X_test) 
print("R2               :", r2_score(y_test, y_pred))
#--copy pasted end

#NEW STUFF
X = dataset.iloc[:, [0,2,5,6] ].values
y = dataset.iloc[:, 3].values

#backward elimination: add constant term in the first column: intercept
X=np.append(arr = np.ones((71,1)).astype(int), values=X, axis=1)
# Constant, Tip, Load, Hardness, Temperature
X_opt=X[:, [0,1,2,3,4]]
#OLS: ordinary least squares, endogenous respose 1-d
regressor=sm.OLS(endog=y, exog=X_opt).fit()
print(regressor.summary() )

#Eliminate Temperature because it has the highest P-value
#  and do again.
# Constant, Tip, Load, Hardness
X_opt=X[:, [0,1,2,3]]
regressor=sm.OLS(endog=y, exog=X_opt).fit()
print(regressor.summary() )

#Eliminate Tip because it has the highest P-value
#  and do again.
# Constant, Tip, Load, Hardness
X_opt=X[:, [0,2,3]]
regressor=sm.OLS(endog=y, exog=X_opt).fit()
print(regressor.summary() )

#NONE OF THE REMAINING HAS A P-VALUE >0.05; 
# none can be eliminated











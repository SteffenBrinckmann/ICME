import pandas as pd  #tables
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('ScratchArea.csv')
"""
#PLOT 1: Steffen's manual version
plt.plot(data['Load'],data['scratch area 1'],'o')
plt.xlabel('Kraft [mN]')
plt.ylabel('Kontaktflaeche [um2]')
plt.show()

#PLOT 2
sns.displot(data['scratch area 1'], bins=10, kde=True)
plt.show()

#PLOT 3
sns.countplot(x="Material", hue="Tip", data=data)
plt.show()

"""
sns.pairplot(data, hue='Material')
plt.show()

sns.heatmap(data.corr(), annot=True, cmap='cividis')
plt.show()

## STEP l: linear regression: exclude material data
# spliting Dataset in Dependent & Independent Variables
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import statsmodels.api as sm

print(data.columns)
X = data.iloc[:, [0,2,5,6] ].values  #skip material, area 2
y = data.iloc[:, 3].values

"""
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0) 

# Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train) 

# Predicting the Test set results
y_pred = regressor.predict(X_test) 
#compare prediction and test data
print("Test       y:",np.round(y_test,2))
print("Prediction y:",np.round(y_pred,2))
print("Relative error:",np.round( np.abs(y_test-y_pred)/y_test*100, 2), '%')
print("Mean square error:", mean_squared_error(y_test, y_pred))
print("R2               :", r2_score(y_test, y_pred))
print("Mean absolute percentage error:", mean_absolute_percentage_error(y_test, y_pred))

"""

#backward elimination: add constant term in the first rows: intercept
XNew=np.append(arr = np.ones((71,1)).astype(int), values=X, axis=1)

#Constant, Tip, Load, Hardness, Temperature
X_opt=XNew[:, [0,1,2,3,4]]
#OLS: ordinary least squares
#endogenous respose 1-d
regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary() )


#Constant, Tip, Load, Hardness
X_opt=XNew[:, [0,1,2,3]]
#OLS: ordinary least squares
#endogenous respose 1-d
regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary() )


#Constant, Load, Hardness
X_opt=XNew[:, [0,2,3]]
#OLS: ordinary least squares
#endogenous respose 1-d
regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary() )

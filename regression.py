import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import statsmodels.api as sm

#always good to check the first rows
dataset= pd.read_csv('ScratchArea.csv') 

#visually check data, first
print(dataset.head())
print('Size:',dataset.shape)
print('Column names:',dataset.columns,'\n')

sns.displot(dataset['scratch area 1'], bins=10, kde=True)
plt.show()

sns.countplot(x="Material", hue="Tip", data=dataset)
plt.show()

#Correlation  chart on different variables for comparision 
# scratch areas are correlated
# with load there is some corralation
# with hardness there is some inverse relation
sns.pairplot(dataset, hue='Material')
plt.show()

# profit split in State level - Looks Florida has the maximum Profit
sns.barplot(x='Material', y='scratch area 1', data=dataset)
plt.show()
sns.lineplot(x='Material',y='scratch area 1',data=dataset)
plt.show()

#gives positive & negative relation between categories
sns.heatmap(dataset.corr(), annot=True, cmap='cividis')
plt.show()

# spread of profit against state 
gridPlot=sns.FacetGrid(dataset[dataset.Material!='Si'], col='Material')
gridPlot.map(sns.kdeplot,'scratch area 1')
plt.show()


##linear regression: exclude material data
# spliting Dataset in Dependent & Independent Variables
X = dataset.iloc[:, [0,2,5,6] ].values  #skip material, area 2
y = dataset.iloc[:, 3].values

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


## Subgroup
subdataset = dataset[np.logical_and(dataset.Material=='Al',dataset.Tip==20)]
X = subdataset.iloc[:, [0,2,5,6] ].values  #skip material, area 2
y = subdataset.iloc[:, 3].values

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



##linear regression: exclude material data
# spliting Dataset in Dependent & Independent Variables
X = dataset.iloc[:, [0,2,5,6] ].values  #skip material, area 2
y = dataset.iloc[:, 3].values

#backward elimination: add constant term in the first rows: intercept
X=np.append(arr = np.ones((71,1)).astype(int), values=X, axis=1)
#print(X)

#Constant, Tip, Material, Load, Harness, Temperature
X_opt=X[:, [0,1,2,3,4]]
#OLS: ordinary least squares
#endogenous respose 1-d
regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary() )
#highest P-value x4=temperature

#step 2
X_opt=X[:, [0,1,2,3]]
regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary() )
#higherst P-value x1=Tip radius

X_opt=X[:, [0,2,3]]
regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary() )
#note cannot exclude more since below  threshold, typically 0.05
#load and hardness have a strong correlation with the scratch area


# https://www.kaggle.com/kavita5/linear-regression-50-startup
# https://www.tutorialandexample.com/linear-regression-tutorial/

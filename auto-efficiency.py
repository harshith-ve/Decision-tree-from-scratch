import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

# Clean the above data by removing redundant columns and rows with junk values
# Compare the performance of your model with the decision tree module from scikit learn


data = data.drop('model year', axis=1)
data = data.drop('car name', axis=1)
data = data.drop('origin', axis=1)
data = data.drop('cylinders', axis=1)
data = data.apply(pd.to_numeric, errors='coerce')
data = data.dropna()


mpg_column = data.pop('mpg')
data['mpg'] = mpg_column


X = data.iloc[:, :-1]
Y = data.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
dataset_train = pd.concat([X_train, Y_train], axis=1)

# Fitting and testing on our decision tree classifier.
tree = DecisionTree(criterion="MSE", max_depth=5)  # Split based on Inf. Gain
tree.fit(X_train, Y_train)
y_hat = tree.predict(X_test)
print("\nRoot Mean Square Error of custom decision tree regressor:", round(rmse(y_hat, Y_test),2))
print("Mean Absolute Error of custom decision tree regressor: ", round(mae(y_hat, Y_test),2),"\n")




# Fitting and testing on our sklear decision tree classifier.

tree_sklearn = DecisionTreeRegressor(max_depth=5, criterion="squared_error")
tree_sklearn.fit(X_train, Y_train)
y_hat_sk = tree_sklearn.predict(X_test)
rmse = np.sqrt(mean_squared_error(Y_test, y_hat_sk))
print(f"Root Mean Square Error of sklearn decision tree regressor: {rmse:.2f}")

# Calculate MAE
mae = mean_absolute_error(Y_test, y_hat_sk)
print(f"Mean Absolute Error of sklearn decision tree regressor: {mae:.2f}")
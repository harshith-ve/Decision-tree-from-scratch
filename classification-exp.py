import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split,cross_val_score, KFold

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)

# Write the code for Q2 a) and b) below. Show your results.
seed = 4
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=seed,stratify=y)

X= pd.DataFrame(X)
y = pd.Series(y, dtype="category")
y_df = pd.DataFrame(y, columns=['new_column_name'])
X = X.join(y_df, rsuffix='_y')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
tree = DecisionTree(criterion="gini_index")
tree.fit(X_train , y_train)
y_hat = tree.predict(X_test)
y_test = y_test.reset_index(drop=True)
print("Accuracy: ", accuracy(pd.Series(y_hat), pd.Series(y_test)))
for cls in y_test.unique():
        recall_value = recall(y_hat, y_test, cls)
        precision_value = precision(y_hat, y_test, cls)
        print(f"Class: {cls}")
        print(f"  Precision: {precision_value}")
        print(f"  Recall: {recall_value}\n")

# Write the code for Q2 a) and b) below. Show your results.

depths = range(1, 11)  # You can adjust the range based on your problem
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
mean_scores = []

for depth in depths:
    # Inner cross-validation loop for hyperparameter tuning
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    tree = DecisionTree(criterion="gini_index", max_depth=depth)
    all_y_test = []
    all_y_pred = []
    for train_index, test_index in inner_cv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)
        y_pred = np.nan_to_num(y_pred, nan=0)
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)
    mean_score = accuracy(pd.Series(all_y_test), pd.Series(all_y_pred))
    mean_scores.append(mean_score)

# Find the depth with the highest mean cross-validation score
optimal_depth = depths[np.argmax(mean_scores)]
print(f"Optimal Depth: {optimal_depth}")
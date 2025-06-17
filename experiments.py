import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 100  # Number of times to run each experiment to calculate the average values


# Function to create fake data (take inspiration from usage.py)
# ...
def generate_fake_data(N, P, case):
    if case == 1:
        X = pd.DataFrame(np.random.randn(N, P))
        y = pd.Series(np.random.randn(N))
    elif case == 2:
        X = pd.DataFrame(np.random.randn(N, P))
        y = pd.Series(np.random.randint(P, size=N), dtype="category")
    elif case == 3:
        X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(P)})
        y = pd.Series(np.random.randint(P, size=N), dtype="category")
    elif case == 4:
        X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(P)})
        y = pd.Series(np.random.randn(N))
    else:
        raise ValueError("Invalid case number")

    return X, y
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
def measure_learning_time(X_train, y_train, case):
    start_time = time.time()
    if (case == 1 or case == 3):
        category = "MSE"
    else:
        category = "gini_index"    
    tree = DecisionTree(criterion=category)
    
    tree.fit(X_train, y_train)
    
    learning_time = time.time() - start_time
    return learning_time, tree

# measuring prediction time
def measure_prediction_time(tree, X_test):
    start_time = time.time()
    
    y_pred = tree.predict(X_test)
    
    prediction_time = time.time() - start_time
    return prediction_time

Ns = [20, 60, 100]
Ms = [10, 40, 70]
cases = [1, 2, 3, 4]
learn_time = []
pred_time = []

for case in cases:
    case_learn_time = []
    case_pred_time = []
    
    for N in Ns:
        for M in Ms:
            X, y = generate_fake_data(N, M, case)
            y_df = pd.DataFrame(y, columns=['new_column_name'])
            X = X.join(y_df, rsuffix='_y')
            
            X_train, X_test = X[:int(0.7*N)], X[int(0.7*N):]
            y_train, y_test = y[:int(0.7*N)], y[int(0.7*N):]
            
            learning_time, trained_tree = measure_learning_time(X_train, y_train, case)
            case_learn_time.append(learning_time)
            
            prediction_time = measure_prediction_time(trained_tree, X_test)
            case_pred_time.append(prediction_time)
    
    learn_time.append(case_learn_time)
    pred_time.append(case_pred_time)
    
    
    
# Function to plot the results
# ...
fig, axes = plt.subplots(nrows=len(cases), ncols=2, figsize=(15, 8))
fig.suptitle('Time Complexity Analysis')

for i, case in enumerate(cases):

    axes[i, 0].set_title(f"Case {case} Learning Time")
    for j, N in enumerate(Ns):
        axes[i, 0].plot(Ms, learn_time[i][j * len(Ms):(j + 1) * len(Ms)], label=f'N={N}')
    axes[i, 0].set_xlabel('P (Features)')
    axes[i, 0].set_ylabel('Time (s)')
    axes[i, 0].legend()

    axes[i, 1].set_title(f"Case {case} Prediction Time")
    for j, N in enumerate(Ns):
        axes[i, 1].plot(Ms, pred_time[i][j * len(Ms):(j + 1) * len(Ms)], label=f'N={N}')
    axes[i, 1].set_xlabel('P (Features)')
    axes[i, 1].set_ylabel('Time (s)')
    axes[i, 1].legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
# Other functions
# ...
# Run the functions, Learn the DTs and Show the results/plots

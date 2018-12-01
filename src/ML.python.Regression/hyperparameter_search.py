import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

train_set = pd.read_csv('../../datasets/housing_train_70.csv')
test_set = pd.read_csv('../../datasets/housing_test_30.csv')


train_y = train_set['median_house_value']
train_X = train_set.drop('median_house_value', axis=1)
test_y = test_set['median_house_value']
test_X = test_set.drop('median_house_value', axis=1)

tree_regressor = DecisionTreeRegressor(random_state=0)

# Create regularization penalty space
max_depth = [20, 50, 70, 110] #numTrees
min_samples_leaf = [3, 4, 6, 10] #minDatapointsInLeafs
max_leaf_nodes = [5, 10, 15, 20, 25] #numLeaves

# Create hyperparameter options
hyperparameters = dict(max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes)

tree_reg_grid = GridSearchCV(tree_regressor, hyperparameters, cv=5, verbose=1, n_jobs=1)
best_model = tree_reg_grid.fit(train_X, train_y)

# View best hyperparameters
print(best_model.best_estimator_)

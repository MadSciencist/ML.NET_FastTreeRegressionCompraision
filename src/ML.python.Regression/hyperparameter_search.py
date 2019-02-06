import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

def main():
    train_set = pd.read_csv('../../datasets/housing_train_70.csv')

    train_y = train_set['median_house_value']
    train_X = train_set.drop('median_house_value', axis=1)

    #numTrees                   max_depth
    #minDatapointsInLeafs        min_samples_leaf
    #numLeaves               max_leaf_nodes

    tree_regressor = DecisionTreeRegressor(random_state=0)

    param_grid = {'max_depth': [5, 10, 20, 50, 100, 200, None],
                  'min_samples_leaf': np.arange(start=1, stop=15, step=2),
                  'max_leaf_nodes': np.arange(start=5, stop=20, step=2)}

    tree_reg_grid = GridSearchCV(tree_regressor, param_grid, cv=10, verbose=1, n_jobs=4)
    best_model = tree_reg_grid.fit(train_X, train_y)

    # display best hyperparameters
    print(best_model.best_estimator_)

if __name__=='__main__':
    main()



    #DecisionTreeRegressor(criterion='mse', max_depth=10, max_features=None,
    #       max_leaf_nodes=19, min_impurity_decrease=0.0,
    #       min_impurity_split=None, min_samples_leaf=1,
    #       min_samples_split=2, min_weight_fraction_leaf=0.0,
    #       presort=False, random_state=0, splitter='best')
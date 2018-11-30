import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

train_set = pd.read_csv('../../datasets/housing_train_70.csv')
test_set = pd.read_csv('../../datasets/housing_test_30.csv')

train_y = train_set['median_house_value']
train_X = train_set.drop('median_house_value', axis=1)
test_y = test_set['median_house_value']
test_X = test_set.drop('median_house_value', axis=1)

tree_regressor = DecisionTreeRegressor(random_state=0)
tree_regressor.fit(train_X, train_y)
score = cross_val_score(tree_regressor, train_X, train_y, cv=5)
print(score.mean())


# evaluate score
y_pred = tree_regressor.predict(test_X)
print("Mean squared error: %.2f" % mean_squared_error(test_y, y_pred))
print('R2: %.4f' % r2_score(test_y, y_pred))
print('\n')


########################################################################3
# TODO
# print some most corelated 
# Plot outputs
# we have to choose some features to plot with comparision to predicted values ?????? check this
#plt.scatter(test_X['median_income'].values, test_y.values,  color='black')
#plt.plot(test_X['median_income'].values, y_pred, color='blue', linewidth=1)

#plt.xticks(())
#plt.yticks(())

#plt.show()
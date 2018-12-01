import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from utils import plot_learning_curve

train_set = pd.read_csv('../../datasets/housing_train_70.csv')
test_set = pd.read_csv('../../datasets/housing_test_30.csv')


train_y = train_set['median_house_value']
train_X = train_set.drop('median_house_value', axis=1)
test_y = test_set['median_house_value']
test_X = test_set.drop('median_house_value', axis=1)

tree_regressor = DecisionTreeRegressor(random_state=0)

#tree_regressor.fit(train_X, train_y)
#score = cross_val_score(tree_regressor, train_X, train_y, cv=5)
#print(score.mean())

## evaluate score
#y_pred = tree_regressor.predict(test_X)
#print("Mean squared error: %.2f" % mean_squared_error(test_y, y_pred))
#print('R2: %.4f' % r2_score(test_y, y_pred))
#print('\n')

#fig, ax = plt.subplots()
#y = test_y.values
#ax.scatter(y, y_pred, edgecolors=(0, 0, 0))
#ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
#ax.set_xlabel('Measured')
#ax.set_ylabel('Predicted')
#plt.show()

#plt.scatter(test_X['housing_median_age'].values, test_y.values,  color='red', s=3)
#plt.scatter(test_X['housing_median_age'].values, y_pred, color='blue', s=3)
#plt.show()

############################################


title = "Learning Curves (Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
plot_learning_curve(estimator=tree_regressor, title="test", X=train_X, y=train_y, ylim=(0.7, 1.01), cv=3, n_jobs=1)

plt.show()
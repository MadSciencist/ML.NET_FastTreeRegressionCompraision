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

tree_regressor.fit(train_X, train_y)
score = cross_val_score(tree_regressor, train_X, train_y, cv=10)
print(f'{score.mean()} += {score.std()}')

# evaluate score
y_pred = tree_regressor.predict(test_X)
print("Mean squared error: %.2f" % mean_squared_error(test_y, y_pred))
print('R2: %.4f' % r2_score(test_y, y_pred))
print('\n')

results = pd.DataFrame(data={'test': test_y, 'pred': y_pred})
results.to_csv(path_or_buf="../../results/python_results.csv", sep=',', index=False)

fig, ax = plt.subplots()
y = test_y.values
ax.scatter(y, y_pred, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
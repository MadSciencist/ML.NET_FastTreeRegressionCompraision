import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
import seaborn as sns

housing = pd.read_csv('../../datasets/housing.csv')
housing_train = pd.read_csv('../../datasets/housing_train_70.csv')
housing_test = pd.read_csv('../../datasets/housing_test_30.csv')

housing.hist(bins=20, figsize=(8, 18))
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)
plt.show()

# correlation matrix
corr_matrix = housing.corr()
# Check the how much each attribute correlates with the median house value
corr_matrix['median_house_value'].sort_values(ascending=False)

attributes = ['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','median_house_value','ocean_proximity']
#scatter_matrix(housing[attributes], figsize=(3,3))
#plt.show()

encoder = LabelBinarizer()
# encode the ocean_proximity column to binary and split it into x-columns, so we wont add fake-weighting
housing = housing.join(pd.DataFrame(encoder.fit_transform(housing['ocean_proximity']), columns=encoder.classes_, index=housing.index))
#drop NaNs
housing = housing.dropna()

#plot heatmap
corr_matrix = housing[attributes].corr()

sns.set(font_scale=1.10)
plt.figure(figsize=(10, 10))
sns.heatmap(corr_matrix, vmax=.8, linewidths=0.01, square = True, annot=True, cmap='icefire', linecolor="white")
plt.title('Correlation between features');
plt.show()

# drop old un-encoded colum
housing = housing.drop('ocean_proximity', axis=1)

########################################################################3

# 30% for testing with some randomness
train_set, test_set = train_test_split(housing, test_size=0.3, random_state=42)

# get our header (first row back (columns=) and create dataframe for easy saving to file
train_set = pd.DataFrame(train_set, columns=housing.columns)
test_set = pd.DataFrame(test_set, columns=housing.columns)

# save both splits to file for further testing (dotnet will use the same files)
train_set.to_csv(path_or_buf="../../datasets/housing_train_70.csv", sep=',', index=False)
test_set.to_csv(path_or_buf="../../datasets/housing_test_30.csv", sep=',', index=False)

lin_reg = LinearRegression()

train_y = train_set['median_house_value']
train_X = train_set.drop('median_house_value', axis=1)

test_y = test_set['median_house_value']
test_X = test_set.drop('median_house_value', axis=1)

lin_reg.fit(train_X, train_y)
y_pred = lin_reg.predict(test_X)

print('Coefficients: \n', lin_reg.coef_)
print("Mean squared error: %.2f" % mean_squared_error(test_y, y_pred))
print('R2: %.4f' % r2_score(test_y, y_pred))
print('\n')


bayes = BayesianRidge()
bayes.fit(train_X, train_y)
y_pred = bayes.predict(test_X)

print('Coefficients: \n', lin_reg.coef_)
print("Mean squared error: %.2f" % mean_squared_error(test_y, y_pred))
print('R2: %.4f' % r2_score(test_y, y_pred))
print('\n')

########################################################################3
# TODO
# print some most corelated 
# Plot outputs
# we have to choose some features to plot with comparision to predicted values ?????? check this
plt.scatter(test_X['median_income'].values, test_y.values,  color='black')
plt.plot(test_X['median_income'].values, y_pred, color='blue', linewidth=1)

plt.xticks(())
plt.yticks(())

plt.show()
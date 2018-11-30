import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import LabelBinarizer
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
scatter_matrix(housing[attributes], figsize=(3,3))
plt.show()

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
plt.title('Correlation between features')
plt.show()

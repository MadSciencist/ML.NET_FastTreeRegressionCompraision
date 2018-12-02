import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns

housing = pd.read_csv('../../datasets/housing.csv')
housing_train = pd.read_csv('../../datasets/housing_train_70.csv')
housing_test = pd.read_csv('../../datasets/housing_test_30.csv')

op_count = housing['ocean_proximity'].value_counts()
plt.figure(figsize=(10,5))
sns.barplot(op_count.index, op_count.values, alpha=0.7)
plt.title('Ocean Proximity Summary')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Ocean Proximity', fontsize=12)
plt.show()

housing.hist(bins=20, figsize=(8, 18))
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)
plt.show()

# correlation matrix
corr_matrix = housing.corr()
# Check the how much each attribute correlates with the median house value
corr_matrix['median_house_value'].sort_values(ascending=False)

attributes = ['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','median_house_value','ocean_proximity']
sm = scatter_matrix(housing[attributes], figsize=(10,10), alpha=0.2)

#Change label rotation
[s.xaxis.label.set_rotation(45) for s in sm.reshape(-1)]
[s.yaxis.label.set_rotation(0) for s in sm.reshape(-1)]

#May need to offset label when rotating to prevent overlap of figure
[s.get_yaxis().set_label_coords(-0.85,0.5) for s in sm.reshape(-1)]
[s.get_xaxis().set_label_coords(-0.15,0) for s in sm.reshape(-1)]

#Hide all ticks
[s.set_xticks(()) for s in sm.reshape(-1)]
[s.set_yticks(()) for s in sm.reshape(-1)]

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

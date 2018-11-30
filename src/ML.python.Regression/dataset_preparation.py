import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

def split_and_save_housing():
    housing = pd.read_csv('../../datasets/housing.csv')

    encoder = LabelBinarizer()
    # encode the ocean_proximity column to binary and split it into x-columns, so we wont add fake-weighting
    housing = housing.join(pd.DataFrame(encoder.fit_transform(housing['ocean_proximity']), columns=encoder.classes_, index=housing.index))

    # drop old un-encoded colum
    housing = housing.drop('ocean_proximity', axis=1)

    # 30% for testing with some randomness
    train_set, test_set = train_test_split(housing, test_size=0.3, random_state=42)

    # get our header (first row back (columns=) and create dataframe for easy saving to file
    train_set = pd.DataFrame(train_set, columns=housing.columns)
    test_set = pd.DataFrame(test_set, columns=housing.columns)

    # save both splits to file for further testing (dotnet will use the same files)
    train_set.to_csv(path_or_buf="../../datasets/housing_train_70.csv", sep=',', index=False)
    test_set.to_csv(path_or_buf="../../datasets/housing_test_30.csv", sep=',', index=False)
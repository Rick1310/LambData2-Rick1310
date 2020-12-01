"""LambData is a collection of DS helper functions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix



data = pd.read_csv("Data/fake_job_postings.csv")

df = data


def df_cleaner(X):
    """Cleans a DF of null values"""
    # TODO - implement
    X = X.copy()

    NullCols = X.isnull()

    for col in NullCols:
        X[col] = X[col].replace(0, np.nan)
        X[col+'_MISSING'] = X[col].isnull()

    return X    

def TrainValTest(df):
    """Performs train test split on a data frame"""
    
    target = 'fraudulent'
    y = df['fraudulent']
    X = df.drop(columns='fraudulent')

    target = 'fraudulent'

    X_train = df[target]
    X_test = df.drop(columns=target, axis=1)
    y_train = df[target]
    y_test = df.drop(columns=target, axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
    
    print (X_train.shape, y_train.shape)
    print (X_test.shape, y_test.shape)
    print('Baseline Accuracy: ', y_train.value_counts(normalize=True).max())

    return TrainValTest

   




  
















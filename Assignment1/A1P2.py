import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def main():
    # import car csv data
    carsdf = pd.read_csv("UsedCars.csv")
    # also keep a numpy copy
    carsnp = carsdf.values

    # split the data set into training and testing
    # this split is seeded
    training, testing = split_data()
    trainnp, testnp = create_train_test_np(training, testing, carsnp)

    return trainnp, testnp


def price_mileage_linreg(trainnp, testnp):
    '''
    runs an ordinary linear regression
    creates scatter plot along with best fit linear regression

    trainnp, testnp - 2D np arrays
    '''
    # extract price and mileage columns from the np arrays
    # we know that price is column 0 and mileage is column 3
    trainnp = trainnp[:, [0, 3]]
    testnp = testnp[:, [0, 3]]

    return trainnp, testnp

def split_data():
    a = np.arange(20063)
    np.random.seed(0)
    np.random.shuffle(a)

    training = a[:15047]
    testing = a[15047:]

    return training, testing


def create_train_test_np(training, testing, carsnp):
    '''
    training, testing - 1D np arrays
    carsnp - 2D np array of car data

    returns:
        trainnp - 2D np array
        testnp - 2D np array
    '''
    trainnp = []
    testnp = []
    
    # create the training np array
    for i in training:
        trainnp.append(carsnp[i])

    for i in testing:
        testnp.append(carsnp[i])

    trainnp = np.array(trainnp)
    testnp = np.array(testnp)

    return trainnp, testnp

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# name of the training data set
CSV_FILE = "Bike_train.csv"

# globals
TESTING_SIZE = 0.25
SHUFFLE = True
NORMALIZE = True

# categorical and continuous features
CATEGORICAL = ['season', 'holiday', 'workingday', 'weather']
CONTINUOUS = ['temp', 'atemp', 'humidity', 'windspeed']

def load_data(csv=CSV_FILE):
    '''
    csv_file (string) - the bike training/testing data set

    note: there may be some extraneous data strucutres here that I never use,
            they can just be removed at the end
    '''
    df = pd.read_csv(csv)
    # headers = list(df.columns)

    y_data = df['count']
    x_data = df.drop(columns=['count'])

    df_np = df.values
    y_np = y_data.values
    x_np = x_data.values

    return y_np, x_np, df


def split_data(y_np, x_np, testing_percent=TESTING_SIZE, shuffle_data=SHUFFLE):
    '''
    takes two np arrays and splits it into a training set and a testing set
    testing_percent - some floating point value less than 1

    note: if data is not enough, may just use all of the data to train and
            cross validate instead of taking out a testing set
    '''
    x_train, x_test, y_train, y_test = train_test_split(x_np, y_np,
                                                test_size=testing_percent,
                                                random_state=20,
                                                shuffle=shuffle_data)

    return x_train, x_test, y_train, y_test


def transform_count(y_np, direction):
    '''
    takes in a np array of the rental counts and transforms it
    direction - either "forward" or "backward"
        "forward" - transforms count to log(count + 1)
        "backward" - transforms back into count numbers
    '''
    if direction == "forward":
        # transform into log(count + 1)
        y_np = y_np + 1
        y_np = np.log(y_np)
        return y_np

    elif direction == "backward":
        # transform back to rental counts
        y_np = np.exp(y_np)
        y_np = y_np - 1
        return y_np

    else:
        print("ERROR: invalid direction argument")


def visualize_train_data():
    '''
    creates plots of different parts of the training data to look for patterns,
    trends, or abnormal data points
    '''
    pass


def clean_data(df, normalize=NORMALIZE):
    '''
    df - pandas dataframe
    normalize - if true will normalize the continuous features only
    perform a variety of operations such as normalizing data and removing data
    '''
    # normalize the continuous features
    if normalize:
        for feature in CONTINUOUS:
            df[feature] = (df[feature] - df[feature].mean()) / \
                                (df[feature].max() - df[feature].min())

    y_data = df['count']
    x_data = df.drop(columns=['count'])
    # I also want to drop the "daylabel" column because that seems useless...
    x_data = x_data.drop(columns=['daylabel'])

    df_np = df.values
    y_np = y_data.values
    x_np = x_data.values

    return y_np, x_np, df

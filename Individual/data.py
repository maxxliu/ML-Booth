# contains all functions related to loading, cleaning/altering data
# Max Liu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# file names
CSV_TRAIN = "hwind/HousingTrain.csv"
CSV_TEST = "hwind/HousingTest.csv"
OUTPUT_TRAIN = "HousingTrain-CLEAN.csv"
OUTPUT_TEST = "HousingTest-CLEAN.csv"

# data splitting parameters
TESTING_SIZE = 0.2
SHUFFLE = True

def clean_train_data(csv=CSV_TRAIN, out_name=OUTPUT_TRAIN):
    '''
    loads a csv file with data and performs cleaning operations as well
    '''
    df = pd.read_csv(csv)

    # we need to find out which columns we need to clean
    # essentially we need to encode the nominal and ordinal data into numbers
    dt = df.dtypes
    headers = df.columns
    object_headers = []
    for h in headers:
        if dt[h] == 'O':
            object_headers.append(h)

    encode_master = []
    # need to go through all of the object type columns and encode integers
    for oh in object_headers:
        print("Encoding for ", oh)
        encode_value = 1
        d = {}
        # now need to loop through the column to encode
        for i, val in df[oh].iteritems():
            if val in d:
                # set the value to whatever number it was encoded to be
                df.at[i, oh] = d[val]
            else:
                # if nan, set to integer 0
                if pd.isnull(val):
                    d[val] = 0
                    df.at[i, oh] = d[val]
                else:
                    # set the encode integer for the value
                    d[val] = encode_value
                    encode_value += 1
                    df.at[i, oh] = d[val]

        encode_master.append(d)
        print(d)
        print()

    df = df.apply(pd.to_numeric)
    df = df.fillna(0)
    df.to_csv(OUTPUT_TRAIN, sep=',', index=False)

    encode_final = {}
    for em in encode_master:
        encode_final = {**encode_final, **em}

    return encode_final


def clean_all_data(csv_train=CSV_TRAIN, csv_test=CSV_TEST):
    '''
    cleans both the training and testing data
    uses the same dictionary keys to encode training and testing data
    '''
    encode_final = clean_train_data()

    df = pd.read_csv(csv_test)

    # we need to find out which columns we need to clean
    # essentially we need to encode the nominal and ordinal data into numbers
    dt = df.dtypes
    headers = df.columns
    object_headers = []
    for h in headers:
        if dt[h] == 'O':
            object_headers.append(h)

    # need to go through all of the object type columns and encode integers
    for oh in object_headers:
        # now need to loop through the column to encode
        for i, val in df[oh].iteritems():
            # hardcoding the missing dictionary values
            if val == "Roll":
                df.at[i, oh] = 6
            elif val == "Membran":
                df.at[i, oh] = 7
            elif val == "PreCast":
                df.at[i, oh] = 16
            elif val == "Other":
                df.at[i, oh] = 17
            else:
                # set the value to whatever number it was encoded to be
                df.at[i, oh] = encode_final[val]

    df = df.apply(pd.to_numeric)
    df = df.fillna(0)
    df.to_csv(OUTPUT_TEST, sep=',', index=False)


def load_data():
    '''
    loads the cleaned trained and testing data
    '''
    # first want to load the training data
    df_train = pd.read_csv(OUTPUT_TRAIN)
    train_y = df_train['SalePrice']
    train_x = df_train.drop(columns=['SalePrice'])
    # confirm that we have removed SalePrice from x training data
    if 'SalePrice' in train_x.columns:
        print('ERROR: SalePrice was not removed from the training data')

    # we also need to transform the
    # load the testing data
    df_test = pd.read_csv(OUTPUT_TEST)

    return train_x, train_y, df_test


def split_data(x_np, y_np, testing_percent=TESTING_SIZE, shuffle_data=SHUFFLE):
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


def log_transform(y_np, direction):
    '''
    takes in a np array of the rental counts and transforms it
    direction - either "forward" or "backward"
        "forward" - transforms count to log(value + 1)
            adding 1 *just in case* anyone sells their home for $0...
        "backward" - transforms back into count numbers
    '''
    if direction == "forward":
        # transform into log(value + 1)
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


#
# # globals
# TESTING_SIZE = 0.25
# SHUFFLE = True
# NORMALIZE = True
#
# # categorical and continuous features
# CATEGORICAL = ['season', 'holiday', 'workingday', 'weather']
# CONTINUOUS = ['temp', 'atemp', 'humidity', 'windspeed']
#
# def load_data(csv=CSV_FILE):
#     '''
#     csv_file (string) - the bike training/testing data set
#
#     note: there may be some extraneous data strucutres here that I never use,
#             they can just be removed at the end
#     '''
#     df = pd.read_csv(csv)
#     # headers = list(df.columns)
#
#     y_data = df['count']
#     x_data = df.drop(columns=['count'])
#
#     df_np = df.values
#     y_np = y_data.values
#     x_np = x_data.values
#
#     return y_np, x_np, df
#
#
# def split_data(y_np, x_np, testing_percent=TESTING_SIZE, shuffle_data=SHUFFLE):
#     '''
#     takes two np arrays and splits it into a training set and a testing set
#     testing_percent - some floating point value less than 1
#
#     note: if data is not enough, may just use all of the data to train and
#             cross validate instead of taking out a testing set
#     '''
#     x_train, x_test, y_train, y_test = train_test_split(x_np, y_np,
#                                                 test_size=testing_percent,
#                                                 random_state=20,
#                                                 shuffle=shuffle_data)
#
#     return x_train, x_test, y_train, y_test
#
#
# def transform_count(y_np, direction):
#     '''
#     takes in a np array of the rental counts and transforms it
#     direction - either "forward" or "backward"
#         "forward" - transforms count to log(count + 1)
#         "backward" - transforms back into count numbers
#     '''
#     if direction == "forward":
#         # transform into log(count + 1)
#         y_np = y_np + 1
#         y_np = np.log(y_np)
#         return y_np
#
#     elif direction == "backward":
#         # transform back to rental counts
#         y_np = np.exp(y_np)
#         y_np = y_np - 1
#         return y_np
#
#     else:
#         print("ERROR: invalid direction argument")
#
#
# def visualize_train_data():
#     '''
#     creates plots of different parts of the training data to look for patterns,
#     trends, or abnormal data points
#     '''
#     pass
#
#
# def clean_data(df, normalize=NORMALIZE):
#     '''
#     df - pandas dataframe
#     normalize - if true will normalize the continuous features only
#     perform a variety of operations such as normalizing data and removing data
#     '''
#     # normalize the continuous features
#     if normalize:
#         for feature in CONTINUOUS:
#             df[feature] = (df[feature] - df[feature].mean()) / \
#                                 (df[feature].max() - df[feature].min())
#
#     y_data = df['count']
#     x_data = df.drop(columns=['count'])
#     # I also want to drop the "daylabel" column because that seems useless...
#     x_data = x_data.drop(columns=['daylabel'])
#
#     df_np = df.values
#     y_np = y_data.values
#     x_np = x_data.values
#
#     return y_np, x_np, df

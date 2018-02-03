from bike_data_clean import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor


def multiple_linear_regression():
    '''
    runs a multiple linear regression over all of the data
    '''
    y_np, x_np, df = load_data()
    x_train, x_test, y_train, y_test = split_data(y_np, x_np)

    # we can first run a straightforward multiple linear regression
    lr_model = LinearRegression()
    lr_model.fit(x_train, y_train)

    # check the performance of this model
    y_predict = lr_model.predict(x_test)
    mse = mean_squared_error(y_predict, y_test)
    r2 = lr_model.score(x_test, y_test)

    print("Performance of multiple linear regression using all features")
    print("Mean Squared Error:  %f" % mse)
    print("RMSE:                %f" % (mse ** 0.5))
    print("R^2:                 %f" % r2)

    #transform the count and try this naive model again
    y_train = transform_count(y_train, "forward")
    lr_model.fit(x_train, y_train)

    #check performance
    y_predict = lr_model.predict(x_test)
    y_predict = transform_count(y_predict, "backward")
    mse = mean_squared_error(y_predict, y_test)
    r2 = lr_model.score(x_test, transform_count(y_test, "forward"))

    print("Performance of multiple linear regression using all features and with transformed rental count")
    print("Mean Squared Error:  %f" % mse)
    print("RMSE:                %f" % (mse ** 0.5))
    print("R^2:                 %f" % r2)


def tree_regression():
    '''
    runs a tree regression over the data, also tries a couple different
    variation of the tree regression to find the best model
    '''
    y_np, x_np, df = load_data()
    x_train, x_test, y_train, y_test = split_data(y_np, x_np)

    # run a normal tree regression over full training set with cross validation
    kFold = 5
    depth = np.arange(2, 25)
    param_grid = {'max_depth': depth}
    tree_grid = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=kFold)

    # THIS IS THE MODEL WITH RAW DATA
    # COMMENTED OUT SO THAT IT WON'T RUN
    # tree_grid.fit(x_train, y_train)
    #
    # # use the best depth to test performance
    # tree_best = tree_grid.best_params_['max_depth']
    # tr_model = DecisionTreeRegressor(max_depth=tree_best)
    # tr_model.fit(x_train, y_train)
    # tree_scores = tree_grid.cv_results_['mean_test_score']
    #
    # y_predict = tr_model.predict(x_test)
    # mse = mean_squared_error(y_predict, y_test)
    # r2 = tr_model.score(x_test, y_test)
    #
    # print("Performance of tree regression using all features")
    # print("Optimal Depth:       %f" % tree_best)
    # print("Mean Squared Error:  %f" % mse)
    # print("RMSE:                %f" % (mse ** 0.5))
    # print("R^2:                 %f" % r2)

    # plot the cross validations results
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(depth, tree_scores, color='blue', linestyle='--', dashes=(5, 2))
    # ax.set_xlabel('depth')
    # ax.set_ylabel('Coefficient of Determination R^2')
    # plt.title("Regression Tree 5-Fold Cross Validation")
    # plt.show()

    # # THIS PART IS THE TRANSFORMED LOG MODEL, IT IS NOT BETTER
    # # transform the count and see if that improves the model
    # y_train = transform_count(y_train, "forward")
    # tree_grid = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=kFold)
    # tree_grid.fit(x_train, y_train)
    #
    # # use the best depth to test performance
    # tree_best = tree_grid.best_params_['max_depth']
    # tr_model = DecisionTreeRegressor(max_depth=tree_best)
    # tr_model.fit(x_train, y_train)
    #
    # y_predict = tr_model.predict(x_test)
    # y_predict = transform_count(y_predict, "backward")
    # mse = mean_squared_error(y_predict, y_test)
    # r2 = tr_model.score(x_test, transform_count(y_test, "forward"))
    #
    # print("Performance of tree regression using all features with transformed rental count")
    # print("Optimal Depth:       %f" % tree_best)
    # print("Mean Squared Error:  %f" % mse)
    # print("RMSE:                %f" % (mse ** 0.5))
    # print("R^2:                 %f" % r2)

    # now we try to do the tree regression after running the data clean
    y_np_c, x_np_c, df_c = clean_data(df)
    x_train, x_test, y_train, y_test = split_data(y_np_c, x_np_c)

    tree_grid = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=kFold)
    tree_grid.fit(x_train, y_train)

    # use the best depth to test performance
    tree_best = tree_grid.best_params_['max_depth']
    tr_model = DecisionTreeRegressor(max_depth=tree_best)
    tr_model.fit(x_train, y_train)
    tree_scores = tree_grid.cv_results_['mean_test_score']

    y_predict = tr_model.predict(x_test)
    mse = mean_squared_error(y_predict, y_test)
    r2 = tr_model.score(x_test, y_test)

    print("Performance of tree regression with removed day labels and normalized")
    print("Optimal Depth:       %f" % tree_best)
    print("Mean Squared Error:  %f" % mse)
    print("RMSE:                %f" % (mse ** 0.5))
    print("R^2:                 %f" % r2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(depth, tree_scores, color='green', linestyle='--', dashes=(5, 2))
    ax.set_xlabel('depth')
    ax.set_ylabel('Coefficient of Determination R^2')
    plt.title("Regression Tree 5-Fold Cross Validation")
    plt.show()

    # I also want to view the actual y against the predicted y
    x = np.arange(len(y_predict))
    plt.plot(x, y_test, color='red', linestyle='--')
    plt.plot(y_predict, color='blue', linestyle='--')
    plt.title("Comparison of actual rental counts and predicted rental counts")
    plt.show()


def adaboost_regression():
    '''
    runs an adaboost regression over the data set
    runs one with just the cleaned data

    we are already certain that cleaned data significantly outperforms
    raw data so we will not waste anymore time training models with raw data

    warning: this takes a REALLY long time to run, would not reccomend running
        this, especially because the results are not amazing
    '''
    # adaboost parameters
    kFold = 5
    param_grid = {'loss': np.array(['linear', 'square', 'exponential']),
                    'learning_rate': np.arange(1, 101, 5)/100,
                    'n_estimators': np.arange(40, 400, 20)}
    adaboost_grid = GridSearchCV(AdaBoostRegressor(), param_grid, cv=kFold)

    # test using the cleaned data
    x_np, y_np, df = load_data()
    y_np_c, x_np_c, df_c = clean_data(df)
    x_train, x_test, y_train, y_test = split_data(y_np_c, x_np_c)
    adaboost_grid.fit(x_train, y_train)
    best_learn = adaboost_grid.best_params_['learning_rate']
    best_loss = adaboost_grid.best_params_['loss']
    best_n = adaboost_grid.best_params_['n_estimators']

    print("Best learning rate: %f" % best_learn)
    print("Best loss function: %s" % best_loss)
    print("Best n estimators: %f" % best_n)

    # train a model using these best parameters
    adaboost_model = AdaBoostRegressor(n_estimators=best_n,
                                        learning_rate=best_learn,
                                        loss=best_loss)
    adaboost_model.fit(x_train, y_train)

    y_predict = adaboost_model.predict(x_test)
    mse = mean_squared_error(y_predict, y_test)
    r2 = adaboost_model.score(x_test, y_test)

    print("Performance of adaboost regression with removed day labels and normalized")
    print("Mean Squared Error:  %f" % mse)
    print("RMSE:                %f" % (mse ** 0.5))
    print("R^2:                 %f" % r2)


def random_forest():
    '''
    runs random forest regression on cleaned data set
    '''
    # random forest parameters
    kFold = 5
    param_grid = {'n_estimators': np.arange(5, 40, 5),
                    'max_features': np.array(['auto', 'sqrt', 'log2']),
                    'max_depth': np.arange(2, 30)}
    forest_grid = GridSearchCV(RandomForestRegressor(), param_grid, cv=kFold)

    # test using the cleaned data
    x_np, y_np, df = load_data()
    y_np_c, x_np_c, df_c = clean_data(df)
    x_train, x_test, y_train, y_test = split_data(y_np_c, x_np_c)
    forest_grid.fit(x_train, y_train)
    best_n = forest_grid.best_params_['n_estimators']
    best_f = forest_grid.best_params_['max_features']
    best_d = forest_grid.best_params_['max_depth']

    print("Best n estimators:   %f" % best_n)
    print("Best max features:   %s" % best_f)
    print("Best max depth:      %f" % best_d)

    # train a model using these best parameters
    forest_model = RandomForestRegressor(n_estimators=best_n,
                                        max_features=best_f,
                                        max_depth=best_d)
    forest_model.fit(x_train, y_train)

    y_predict = forest_model.predict(x_test)
    mse = mean_squared_error(y_predict, y_test)
    r2 = forest_model.score(x_test, y_test)

    print("Performance of random forest regression with removed day labels and normalized")
    print("Mean Squared Error:  %f" % mse)
    print("RMSE:                %f" % (mse ** 0.5))
    print("R^2:                 %f" % r2)


def final_model():
    '''
    train the final chosen model using all of the training data
    take those optimal parameters and predict the values of the test set
    put the predicted values into a csv file
    done.
    '''
    # final chosen model is random forest regressor
    # random forest parameters
    kFold = 5
    param_grid = {'n_estimators': np.arange(5, 40, 5),
                    'max_features': np.array(['auto', 'sqrt', 'log2']),
                    'max_depth': np.arange(2, 30)}
    forest_grid = GridSearchCV(RandomForestRegressor(), param_grid, cv=kFold)

    # train using the all of the training cleaned data
    y_np, x_np, df = load_data()
    y_np_c, x_np_c, df_c = clean_data(df)

    forest_grid.fit(x_np_c, y_np_c)
    best_n = forest_grid.best_params_['n_estimators']
    best_f = forest_grid.best_params_['max_features']
    best_d = forest_grid.best_params_['max_depth']

    print("Best n estimators:   %f" % best_n)
    print("Best max features:   %s" % best_f)
    print("Best max depth:      %f" % best_d)

    # train a model using these best parameters
    forest_model = RandomForestRegressor(n_estimators=best_n,
                                        max_features=best_f,
                                        max_depth=best_d)
    forest_model.fit(x_np_c, y_np_c)

    # import the test dataset
    df_test = pd.read_csv("bike_test.csv")
    # clean the data, the clean data function doesnt work for this dataframe
    for feature in CONTINUOUS:
        df_test[feature] = (df_test[feature] - df_test[feature].mean()) / \
                        (df_test[feature].max() - df_test[feature].min())
    df_test = df_test.drop(columns=['daylabel'])
    df_test_np = df_test.values

    # predict the values using our trained model
    y_predict = forest_model.predict(df_test_np)

    np.savetxt("hw2-1-maxliu.csv", y_predict, delimiter=",")

    return df_test, df_test_np

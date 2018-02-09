from data import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.svm import SVR

def run_models():
    '''
    runs the different models
    '''
    # tree_regression()
    # print("_______________")
    # random_forest()
    # print("_______________")
    # adaboost_regression()
    # print("_______________")
    # svm_regression()
    # print("_______________")
    final_model()


def tree_regression():
    '''
    runs a tree regression over the data
    '''
    print("Currently running Tree Regression")
    train_x, train_y, _ = load_data()
    # convert to np array
    train_x = train_x.values
    train_y = train_y.values
    # convert the y values to log
    train_y = log_transform(train_y, "forward")
    # split the data
    x_train, x_test, y_train, y_test = split_data(train_x, train_y)

    # run a normal tree regression over full training set with cross validation
    # tree regression parameters
    kFold = 5
    depth = np.arange(2, 25)
    param_grid = {'max_depth': depth}

    # test using training data
    tree_grid = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=kFold)
    tree_grid.fit(x_train, y_train)
    tree_best = tree_grid.best_params_['max_depth']
    print("Optimal Depth:       %f" % tree_best)

    # use the best depth to test performance
    tr_model = DecisionTreeRegressor(max_depth=tree_best)
    tr_model.fit(x_train, y_train)
    # tree_scores = tree_grid.cv_results_['mean_test_score']

    y_predict = tr_model.predict(x_test)
    mse = mean_squared_error(y_predict, y_test)
    r2 = tr_model.score(x_test, y_test)

    print("Performance of tree regression")
    print("Mean Squared Error:  %f" % mse)
    print("RMSE:                %f" % (mse ** 0.5))
    print("R^2:                 %f" % r2)


def adaboost_regression():
    '''
    runs an adaboost regression over the data set
    warning: this will probably take a long time to run
    '''
    print("Currently running AdaBoost Regression")
    train_x, train_y, _ = load_data()
    # convert to np array
    train_x = train_x.values
    train_y = train_y.values
    # convert the y values to log
    train_y = log_transform(train_y, "forward")
    # split the data
    x_train, x_test, y_train, y_test = split_data(train_x, train_y)

    # adaboost parameters
    kFold = 5
    param_grid = {'loss': np.array(['linear', 'square', 'exponential']),
                    'learning_rate': np.arange(1, 101, 5)/100,
                    'n_estimators': np.arange(40, 400, 20)}
    adaboost_grid = GridSearchCV(AdaBoostRegressor(), param_grid, cv=kFold)

    # test using the training data
    adaboost_grid.fit(x_train, y_train)
    best_learn = adaboost_grid.best_params_['learning_rate']
    best_loss = adaboost_grid.best_params_['loss']
    best_n = adaboost_grid.best_params_['n_estimators']

    print("Best learning rate:  %f" % best_learn)
    print("Best loss function:  %s" % best_loss)
    print("Best n estimators:   %f" % best_n)

    # train a model using these best parameters
    adaboost_model = AdaBoostRegressor(n_estimators=best_n,
                                        learning_rate=best_learn,
                                        loss=best_loss)
    adaboost_model.fit(x_train, y_train)

    y_predict = adaboost_model.predict(x_test)
    mse = mean_squared_error(y_predict, y_test)
    r2 = adaboost_model.score(x_test, y_test)

    print("Performance of adaboost regression")
    print("Mean Squared Error:  %f" % mse)
    print("RMSE:                %f" % (mse ** 0.5))
    print("R^2:                 %f" % r2)


def random_forest():
    '''
    runs random forest regression on cleaned data set
    '''
    print("Currently running Random Forest Regression")
    train_x, train_y, _ = load_data()
    # convert to np array
    train_x = train_x.values
    train_y = train_y.values
    # convert the y values to log
    train_y = log_transform(train_y, "forward")
    # split the data
    x_train, x_test, y_train, y_test = split_data(train_x, train_y)

    # random forest parameters
    kFold = 5
    param_grid = {'n_estimators': np.arange(200, 900, 100),
                    'max_features': np.array(['auto', 'sqrt', 'log2']),
                    'max_depth': np.arange(2, 30)}
    forest_grid = GridSearchCV(RandomForestRegressor(), param_grid, cv=kFold)

    # test using training data
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

    print("Performance of random forest regression")
    print("Mean Squared Error:  %f" % mse)
    print("RMSE:                %f" % (mse ** 0.5))
    print("R^2:                 %f" % r2)


def svm_regression():
    '''
    runs a support vector regression
    '''
    print("Currently running Support Vector Regression")
    train_x, train_y, _ = load_data()
    # convert to np array
    train_x = train_x.values
    train_y = train_y.values
    # convert the y values to log
    train_y = log_transform(train_y, "forward")
    # split the data
    x_train, x_test, y_train, y_test = split_data(train_x, train_y)

    # support vector regression
    kFold = 5
    param_grid = {'C': np.arange(700, 1200, 100),
                    'epsilon': np.arange(0.1, 0.30, 0.05),
                    'kernel': ['rbf']}
    svr_grid = GridSearchCV(SVR(), param_grid, cv=kFold)

    # test using training data
    svr_grid.fit(x_train, y_train)
    best_c = svr_grid.best_params_['C']
    best_e = svr_grid.best_params_['epsilon']
    best_k = svr_grid.best_params_['kernel']

    print("Best C:          %f" % best_c)
    print("Best epsilon:    %f" % best_e)
    print("Best kernel:     %s" % best_k)

    # train a model using these best parameters
    svr_model = SVR(C=best_c, epsilon=best_e, kernel=best_k)
    svr_model.fit(x_train, y_train)

    y_predict = svr_model.predict(x_test)
    mse = mean_squared_error(y_predict, y_test)
    r2 = svr_model.score(x_test, y_test)

    print("Performance of support vector regression")
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
    train_x, train_y, test_x = load_data()
    # convert to np array
    train_x = train_x.values
    train_y = train_y.values
    test_x = test_x.values
    # convert the y values to log
    train_y = log_transform(train_y, "forward")

    # want to double check the shape of train_x and test_x are the same
    if test_x.shape[1] == train_x.shape[1]:
        print("Correct size")
    else:
        print("ERROR: sizing is wrong")
        return

    # random forest regressor is our best model
    # random forest parameters
    kFold = 5
    param_grid = {'n_estimators': np.arange(600, 900, 100),
                    'max_features': np.array(['sqrt'])}
    forest_grid = GridSearchCV(RandomForestRegressor(), param_grid, cv=kFold)

    print("Currently running final model and creating csv")
    # test using training data
    forest_grid.fit(train_x, train_y)
    best_n = forest_grid.best_params_['n_estimators']
    best_f = forest_grid.best_params_['max_features']

    print("Best n estimators:   %f" % best_n)
    print("Best max features:   %s" % best_f)

    # train a model using these best parameters
    forest_model = RandomForestRegressor(n_estimators=best_n,
                                        max_features=best_f)
    forest_model.fit(train_x, train_y)

    y_predict = forest_model.predict(test_x)
    y_predict = log_transform(y_predict, "backward")

    np.savetxt("hwind-maxliu.csv", y_predict, delimiter=",")


if __name__ == '__main__':
    run_models()

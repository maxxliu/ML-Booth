download.file(
    paste("https://raw.githubusercontent.com/ChicagoBoothML/MLClassData/master/",
          "HumanActivityRecognitionUsingSmartphones/ParseData.R",sep=""),
    "ParseData.R")
source("ParseData.R")

# load data
data = parse_human_activity_recog_data()

###### fun starts here

library(h2o)

# start h2o server
h2oServer = h2o.init(nthreads = -1)

# load data into h2o format
Xtrain = as.h2o(data$X_train, destination_frame = "Xtrain")
Ytrain = as.h2o(data$y_train, destination_frame = "Ytrain")
Xtest = as.h2o(data$X_test, destination_frame = "Xtest")
Ytest = as.h2o(data$y_test, destination_frame = "Ytest")

# train a simple neural network
p = ncol(Xtrain)
simpleNN = h2o.deeplearning(x=1:p, y=p+1,     # specify which columns are features and which are target
                            training_frame = h2o.cbind(Xtrain, Ytrain),   # combine features with labels
                            hidden = 10,      # 1 hidden layer with 10 neurons
                            epochs = 5,       # this is a test run, so 5 epochs is fine
                            model_id = "simple_nn_model"
                            )

phat = h2o.predict(simpleNN, Xtest) # compute probabilities for new data 

o = h2o.confusionMatrix(simpleNN, h2o.cbind(Xtest, Ytest)) # compute confusion matrix
names(o)
o$Error

# not bad, but can we do better???????

# simple model worked.... maybe try something deeper
simpleDL = h2o.deeplearning(x=1:p, y=p+1,     # specify which columns are features and which are target
                            training_frame = h2o.cbind(Xtrain, Ytrain),   # combine features with labels
                            hidden = c(50, 20),      # 2 hidden layers with 50 and 20 neurons 
                            epochs = 5,       # this is a test run, so 5 epochs is fine
                            l1 = 1e-5,          # regularize
                            model_id = "simple_dl_model"
                            )
  
h2o.confusionMatrix(simpleDL, h2o.cbind(Xtest, Ytest)) # compute confusion matrix

# hmmmm.... not much improvement, can you help me and find a better model????


### here are some tree models

# random forest
rf.model = h2o.randomForest(x=1:p, y=p+1,     # specify which columns are features and which are target
                            training_frame = h2o.cbind(Xtrain, Ytrain),   # combine features with labels
                            ntrees = 5,                                   # this poor forest only has 5 trees
                            min_rows = 20,                                # each leaf needs to have at least 20 nodes
                            max_depth = 10,                               # we do not want too deep trees
                            model_id = "simple_rf_model"
                            )

# random forest
rf.model = h2o.randomForest(x=1:p, y=p+1,     # specify which columns are features and which are target
                            training_frame = h2o.cbind(Xtrain, Ytrain),   # combine features with labels
                            ntrees = 50,                                  # this poor forest only has 5 trees
                            min_rows = 20,                                # each leaf needs to have at least 20 nodes
                            max_depth = 10,                               # we do not want too deep trees
                            model_id = "simple_rf_model",
                            nfold = 5
                            )

h2o.confusionMatrix(rf.model, h2o.cbind(Xtest, Ytest)) # compute confusion matrix

# boosting

gbm.model =  h2o.gbm(x=1:p, y=p+1,     # specify which columns are features and which are target
                     training_frame = h2o.cbind(Xtrain, Ytrain),   # combine features with labels
                     ntrees = 5,                                   # this poor boosting model only has 5 trees
                     min_rows = 20,                                # each leaf needs to have at least 20 nodes
                     max_depth = 3,                                # we want shallow trees
                     model_id = "simple_gbm_model"
                     )

h2o.confusionMatrix(gbm.model, h2o.cbind(Xtest, Ytest)) # compute confusion matrix



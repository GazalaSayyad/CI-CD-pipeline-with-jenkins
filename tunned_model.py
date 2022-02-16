
import os
import numpy as np
from sklearn.neural_network import MLPClassifier
import pandas as pd
from joblib import dump, load
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV


class TUNModel:


    __training__ = "./dataset/train.csv"
    __filename_tuned_nn__ = './model/tuned_nn.joblib'

 
    def __init__(self):
        pass

    def tuned_model(self):

        training = "./dataset/train.csv"
            
        data_train = pd.read_csv(TUNModel.__training__)
            
        y_train = data_train['# Letter'].values
        X_train = data_train.drop(data_train.loc[:, 'Line':'# Letter'].columns, axis = 1)

        print("Shape of the training data")
        print(X_train.shape)
        print(y_train.shape)
            
        # Data normalization (0,1)
        X_train = preprocessing.normalize(X_train, norm='l2')


        # Models training
        # Neural Networks multi-layer perceptron (MLP) algorithm that trains using Backpropagation
        param_grid = [
        {
            'activation' : ['identity', 'logistic', 'tanh', 'relu'],
            'solver' : ['lbfgs', 'sgd', 'adam'],
            'hidden_layer_sizes': [(300,),(500,)],
            'max_iter': [20000],
            'alpha': [1e-5, 0.001, 0.01, 0.1, 1, 10],
            'random_state':[1]
        }
        ]

        clf_neuralnet = GridSearchCV(MLPClassifier(), param_grid,scoring='accuracy')
        clf_neuralnet.fit(X_train, y_train)
        print("The Neural Net (few parameters) best prediction is ...")
        print("Tunned NN model score: %f" %clf_neuralnet.score(X_test, y_test))
        print("Best parameters set found on development set:")
        print(clf_neuralnet.best_params_)


if __name__ == '__main__':
    tn = TUNModel()
    tn.tuned_model()

import os
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import pandas as pd
import joblib
from joblib import dump, load
from sklearn import preprocessing


class ModelNN:


    __training__ = "./dataset/train.csv"
    __testing__ = "./dataset/test.csv"
    __filename_nn__ = "./model/nn.joblib"

 
    def __init__(self):
        pass


    def nn_model(self):
        training = "./dataset/train.csv"
    
        data_train = pd.read_csv(ModelNN.__training__)
            
        y_train = data_train['# Letter'].values
        X_train = data_train.drop(data_train.loc[:, 'Line':'# Letter'].columns, axis = 1)

        print("Shape of the training data")
        print(X_train.shape)
        print(y_train.shape)
            
        # Data normalization (0,1)
        X_train = preprocessing.normalize(X_train, norm='l2')
        
        # Models training
        clf_NN = MLPClassifier(solver='adam', activation='relu', alpha=0.0001, hidden_layer_sizes=(500,), random_state=0, max_iter=1000)
        clf_NN.fit(X_train, y_train)
        print("Model fitted..........")
        dump(clf_NN,ModelNN.__filename_nn__)

    def check_nn_score(self):
    # Load, read and normalize testing data
        data_test = pd.read_csv(ModelNN.__testing__)

        y_test = data_test['# Letter'].values
        X_test = data_test.drop(data_test.loc[:, 'Line':'# Letter'].columns, axis = 1)

        # Data normalization (0,1)
        X_test = preprocessing.normalize(X_test, norm='l2')

        # load and Run model
        clf_NN = load(ModelNN.__filename_nn__)

        print("NN model score: %f" %clf_NN.score(X_test, y_test))
        
            

if __name__ == '__main__':
    nn = ModelNN()
    nn.nn_model()
    nn.check_nn_score()
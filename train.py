

import os
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import pandas as pd
import joblib
from joblib import dump, load
from sklearn import preprocessing


class Model:


    __training__ = "./dataset/train.csv"
    __testing__ = "./dataset/test.csv"
    __filename_lda__ = './model/model.joblib'
   

 
    def __init__(self):
        pass

    def lda_model(self):
        # Load, read and normalize training data
        training = "./dataset/train.csv"
    
        data_train = pd.read_csv(Model.__training__)
            
        y_train = data_train['# Letter'].values
        X_train = data_train.drop(data_train.loc[:, 'Line':'# Letter'].columns, axis = 1)

        print("Shape of the training data")
        print(X_train.shape)
        print(y_train.shape)
            
        # Data normalization (0,1)
        X_train = preprocessing.normalize(X_train, norm='l2')
        
        # Models training
        
        # Linear Discrimant Analysis (Default parameters)
        clf_lda = LinearDiscriminantAnalysis()
        clf_lda.fit(X_train, y_train)
        
        # Serialize model
        dump(clf_lda, Model.__filename_lda__)

    def check_score(self):
    # Load, read and normalize testing data
        data_test = pd.read_csv(Model.__testing__)

        y_test = data_test['# Letter'].values
        X_test = data_test.drop(data_test.loc[:, 'Line':'# Letter'].columns, axis = 1)

        # Data normalization (0,1)
        X_test = preprocessing.normalize(X_test, norm='l2')

        # load and Run model
        clf_lda = load(Model.__filename_lda__)
        print("LDA model score: %f" %clf_lda.score(X_test, y_test))
        
               
            
if __name__ == '__main__':
    tm = Model()
    tm.lda_model()
    tm.check_score()
 
   
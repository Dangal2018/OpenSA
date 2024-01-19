"""
    -*- coding: utf-8 -*-
    @Time   :2022/04/12 17:10
    @Author : Pengyou FU
    @blogs  : https://blog.csdn.net/Echo_Code?spm=1000.2115.3001.5343
    @github : https://github.com/FuSiry/OpenSA
    @WeChat : Fu_siry
    @License：Apache-2.0 license

"""

from Regression.ClassicRgs import Pls, Anngression, Svregression, ELM
from Regression.CNN import CNNTrain
from sklearn.model_selection import GridSearchCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
import numpy as np
params_grid = {
    'n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #'n_fold': [2, 3, 4, 5, 6, 7, 8, 9, 10],
}

cv = LeaveOneOut()


def QuantitativeAnalysis(model, X_train, X_test, y_train, y_test):
    if model == "Pls":
        pls = PLSRegression()
        grid_search = GridSearchCV(pls, params_grid, cv=cv, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        # print("cv_results_: ", grid_search.cv_results_)
        print("Best Parameters: ", grid_search.best_params_)
        print("Best RMSECV: ", (-grid_search.best_score_) ** 0.5)

        n_components = grid_search.best_params_['n_components']
        plt.figure(figsize=(8, 6))

        plt.plot(grid_search.cv_results_['param_n_components'], -grid_search.cv_results_['mean_test_score'], 'o-')

        plt.title('Scree Plot')
        plt.xlabel('Number of Components')
        # plt.ylabel('Explained Variance')
        plt.show()

        Rmse, R2, Mae = Pls(X_train, X_test, y_train, y_test,11)#grid_search.best_params_['n_components'])
    elif model == "ANN":
        Rmse, R2, Mae = Anngression(X_train, X_test, y_train, y_test)
    elif model == "SVR":
        Rmse, R2, Mae = Svregression(X_train, X_test, y_train, y_test)
    elif model == "ELM":
        Rmse, R2, Mae = ELM(X_train, X_test, y_train, y_test)
    elif model == "CNN":
        Rmse, R2, Mae = CNNTrain("AlexNet", X_train, X_test, y_train, y_test, 150)
    else:
        print("no this model of QuantitativeAnalysis")

    return Rmse, R2, Mae

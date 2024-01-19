import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor

# import hpelm

"""
    -*- coding: utf-8 -*-
    @Time   :2022/04/12 17:10
    @Author : Pengyou FU
    @blogs  : https://blog.csdn.net/Echo_Code?spm=1000.2115.3001.5343
    @github : https://github.com/FuSiry/OpenSA
    @WeChat : Fu_siry
    @License：Apache-2.0 license

"""

from sklearn.svm import SVR
from Evaluate.RgsEvaluate import ModelRgsevaluate
from ModelSL.ModelSL import save_model, load_model
import matplotlib.pyplot as plt


def Pls(X_train, X_test, y_train, y_test, N=4):
    model = PLSRegression(n_components=N)
    # fit the model
    model.fit(X_train, y_train)

    save_model(model)

    y_train_pred = model.predict(X_train)

    plt.rcParams['font.sans-serif'] = ['times new roman']
    plt.rcParams['axes.unicode_minus'] = False
    ax = plt.gca()
    plt.xlabel('True Content')
    plt.ylabel('Predicted Content')
    # plt.title('Results', fontsize=22, fontweight='semibold')
    # x_col = np.linspace(1, len(y_train), len(y_train))
    # plt.scatter(x_col, y_train, color='red', label='True')
    plt.scatter(y_train, y_train_pred, color='red', label='Predicted')
    # 趋势线
    z1 = np.polyfit(y_train, y_train_pred, 1)
    p1 = np.poly1d(z1)
    plt.plot(y_train, p1(y_train), "--", label='Trend line')
    plt.legend(loc='best')

    y_pred = model.predict(X_test)
    rmse, r2, mae = ModelRgsevaluate(y_train_pred, y_train)
    ax.text(0.9, 0.5, 'RMSECV=%.3f' % rmse, ha='right', va='center', transform=ax.transAxes, fontsize=10)
    ax.text(0.9, 0.6, 'R2=%.3f' % r2, ha='right', va='center', transform=ax.transAxes, fontsize=10)
    plt.figure(figsize=(6, 4))
    x_col = np.linspace(1, len(y_test), len(y_test))
    y_test = np.transpose(y_test)
    # print(x_col.shape, y_test.shape)
    # plt.xticks(range(1, len(y_test) + 1, 1))
    ay = plt.gca()

    # ay.set_xlim(0, len(y_test) + 1)
    # ax.set_ylim(50, 65)
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.scatter(x_col, y_test, color='red', label='True')
    # plt.scatter(x_col, y_pred, color='blue', label='Predicted')
    plt.scatter(y_test, y_pred, color='red', label='Predicted')

    plt.xlabel('True Content')
    plt.ylabel('Predicted Content')
    # plt.title('Results', fontsize=22)

    # 趋势线
    z2 = np.polyfit(y_test, y_pred, 1)
    p2 = np.poly1d(z2)
    plt.plot(y_test, p2(y_test), "b--", label='trend line')

    plt.legend(loc='best')
    # 画训练图

    print("trainset-The RMSE:{} R2:{}, MAE:{} of result!".format(rmse, r2, mae))
    Rmse, R2, Mae = ModelRgsevaluate(y_pred, y_test)

    ay.text(0.9, 0.5, 'RMSEP=%.3f' % Rmse, ha='right', va='center', transform=ay.transAxes, fontsize=10)
    ay.text(0.9, 0.6, 'R2=%.3f' % R2, ha='right', va='center', transform=ay.transAxes, fontsize=10)
    plt.show()
    print("testset-The RMSE:{} R2:{}, MAE:{} of result!".format(Rmse, R2, Mae))
    print("testset results:")
    for i in range(0, len(y_pred)):
        print(y_test[i], y_pred[i])

    # datapath = ".//Data//0110hydro.csv"
    #     #
    #     # data1 = np.loadtxt(open(datapath, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
    #     # data = data1[:, :-3]
    #     # label = data1[:, -3]
    #     #
    #     # y_pred = model.predict(data)
    #     # print("predict result:")
    #     # for (i, j) in zip(y_pred, label):
    #     #     print(i, j)

    return Rmse, R2, Mae


def Svregression(X_train, X_test, y_train, y_test):
    model = SVR(C=2, gamma=1e-07, kernel='linear')
    model.fit(X_train, y_train)

    # predict the values
    y_pred = model.predict(X_test)
    Rmse, R2, Mae = ModelRgsevaluate(y_pred, y_test)

    return Rmse, R2, Mae


def Anngression(X_train, X_test, y_train, y_test):
    model = MLPRegressor(
        hidden_layer_sizes=(20, 20), activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
        learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=400, shuffle=True,
        random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
        early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.fit(X_train, y_train)

    # predict the values
    y_pred = model.predict(X_test)
    Rmse, R2, Mae = ModelRgsevaluate(y_pred, y_test)

    return Rmse, R2, Mae


def ELM(X_train, X_test, y_train, y_test):
    model = hpelm.ELM(X_train.shape[1], 1)
    model.add_neurons(20, 'sigm')

    model.train(X_train, y_train, 'r')
    y_pred = model.predict(X_test)

    Rmse, R2, Mae = ModelRgsevaluate(y_pred, y_test)

    return Rmse, R2, Mae

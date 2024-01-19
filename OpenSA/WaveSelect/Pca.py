"""
    -*- coding: utf-8 -*-
    @Time   :2022/04/12 17:10
    @Author : Pengyou FU
    @blogs  : https://blog.csdn.net/Echo_Code?spm=1000.2115.3001.5343
    @github : https://github.com/FuSiry/OpenSA
    @WeChat : Fu_siry
    @License：Apache-2.0 license

"""
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np




def Pca(X, nums=5):
    """
       :param X: raw spectrum data, shape (n_samples, n_features)
       :param nums: Number of principal components retained
       :return: X_reduction：Spectral data after dimensionality reduction
    """
    var = []
    pca = PCA(n_components=nums)  # 保留的特征数码
    pca.fit(X)
    var.append(pca.explained_variance_ratio_)
    var = np.array(var)
    #StandardScaler
    # plt.errorbar(np.linspace(1, X.shape[1], X.shape[1]), np.mean(var, axis=0), yerr=np.std(var, axis=0),
    #              lw=2, elinewidth=1.5, ms=5, capsize=3, fmt='b-o')
    plt.show()
    X_reduction = pca.transform(X)

    return X_reduction

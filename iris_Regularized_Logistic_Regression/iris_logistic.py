# -*- coding: utf-8 -*-
"""
Created on Oct 21 2018
@author:  秦祥翔
"""

from sklearn import datasets
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pca_my_test as pm
import numpy as np
import scipy.io as scio 

def sig(Z):
    return 1.0 / (1 + np.exp(-Z))

def nor_logistic(X, Y, k):
    m, n = X.shape
    alpha = 0.5  # 学习率
    lamb = 1  # 惩罚项系数
    b = 0
    W = np.random.rand(n, 1)
    # W = np.ones((n, 1))
    # 梯度下降
    for i in range(k):
        pre_Y = sigmoid(np.dot(W.T, X.T) + b)
        err = Y.T - pre_Y  # 误差err
        L = -(np.dot(Y, np.log(pre_Y)) + np.dot(1 - Y, np.log(1 - pre_Y))) + lamb*np.sum(np.power(W, 2))/(2*m)  # 正则化，添加惩罚项
        loss = np.sum(L) / m  # loss
        dW = np.dot(X.T, err.T) / m  # 更新参数
        db = np.sum(err) / m
        print(loss)
        W += alpha * dW
        b += alpha * db
    return W, b
    
def result_show(data, labels, W1, b1, W2, b2):
    xMat = np.array(data)
    label = np.array(labels)
    n, m = xMat.shape  # 取其行数，即数据点的个数
    type0_x = []  # label 为 0
    type0_y = []
    type0_z = []
    type1_x = []  # label 为 1
    type1_y = []
    type1_z = []
    type2_x = []  # label 为 2
    type2_y = []
    type2_z = []
    for i in range(n):
        if label[i] == 0:  # 标签0
            type0_x.append(xMat[i][0])
            type0_y.append(xMat[i][1])
            if m >= 3:
                type0_z.append(xMat[i][2])

        elif label[i] == 1:  # 标签1
            type1_x.append(xMat[i][0])
            type1_y.append(xMat[i][1])
            if m >= 3:
                type1_z.append(xMat[i][2])

        else:  # 标签2
            type2_x.append(xMat[i][0])
            type2_y.append(xMat[i][1])
            if m >= 3:
                type2_z.append(xMat[i][2])

    if m >= 3:
        # 如果数据维度超过3, 将参数存储下来，使用matlab进行三维隐函数绘图
        scio.savemat('.data/W1.mat', {'W1': W1})
        scio.savemat('.data/b1.mat', {'b1': b1})
        scio.savemat('.data/W2.mat', {'W2': W2})
        scio.savemat('.data/b2.mat', {'b2': b2})
        print("============参数已存储为.mat文件==============")
        print("因使用了非线性的特征，本人水平有限，无法在python中绘制三维隐函数的图像 这里需要使用matlab绘图")
    else:
        # 绘制2d决策边界图
        plt.figure()
        axes = plt.subplot(111)

        x = np.linspace(-3, 3, 500)
        y = np.linspace(-3, 3, 500)

        x2 = np.linspace(0, 3, 500)
        y2 = np.linspace(-3, 3, 500)

        x, y = np.meshgrid(x, y)
        x2, y2 = np.meshgrid(x2, y2)
        z01 = x * W1[0, 0] + y * W1[1, 0] + W1[2, 0] * x ** 2 + W1[3, 0] * y ** 2 + W1[4, 0] * x * y + b1
        z12 = x2 * W2[0, 0] + y2 * W2[1, 0] + W2[2, 0] * x2 ** 2 + W2[3, 0] * y2 ** 2 + W2[4, 0] * x2 * y2 + b2
        plt.contour(x, y, z01, 0)
        plt.contour(x2, y2, z12, 0)
        type0 = axes.scatter(type0_x, type0_y, s=20, c='red', marker='*')
        type1 = axes.scatter(type1_x, type1_y, s=20, c='green', marker='x')
        type2 = axes.scatter(type2_x, type2_y, s=20, c='blue', marker='o')
        axes.set_ylabel('X[2]')
        axes.set_xlabel('X[1]')
        axes.legend((type0, type1, type2), (u'label0', u'label1', u'label2'), loc=2)
        plt.show()


if __name__ == '__main__':
    
    iris = datasets.load_iris()
    
    X = iris.data
    Y = iris.target.reshape(-1, 1)
    
    k = int(input("PCA后的数据维度："))
    
    traindata, recodata = pm.datapca(X,k)
    traindata = np.array(traindata)

    n, m = traindata.shape
    if m == 3:
        traindata_PP = np.hstack((traindata, traindata[:, 0].reshape((-1, 1))**2,
                               traindata[:, 1].reshape((-1, 1))**2,
                               traindata[:, 2].reshape((-1, 1))**2,
                               traindata[:, 0].reshape((-1, 1))*traindata[:, 1].reshape((-1,1)),
                               traindata[:, 1].reshape((-1, 1))*traindata[:, 2].reshape((-1,1)),
                               traindata[:, 0].reshape((-1, 1))*traindata[:, 2].reshape((-1,1)),
                               )) # 添加非线性特征
    elif m==2:
        traindata_PP = np.hstack((traindata, traindata[:, 0].reshape((-1, 1)) ** 2,
                                  traindata[:, 1].reshape((-1, 1)) ** 2,
                                  traindata[:, 0].reshape((-1, 1)) * traindata[:, 1].reshape((-1, 1))
                                  ))  # 添加非线性特征
    
    n = int(input("迭代次数："))
    W1, b1 = gradA(traindata_PP[:100], Y[:100], n)
    W2, b2 = gradA(traindata_PP[50:150], Y[50:150] - 1, n)
    print(W1, b1, '\n', W2, b2)
    result_show(traindata, iris.target, W1, b1, W2, b2)

 

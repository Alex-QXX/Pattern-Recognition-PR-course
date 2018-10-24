# -*- coding: utf-8 -*-
"""
Created on Oct 21 2018
@author:  秦祥翔
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer,scale,normalize,MinMaxScaler


def load_data():
    """
    Criteo 广告数据读取
    :return: 训练集的特征和标签，测试集的ID和特征
    """
    train_data = pd.read_csv('./data/train.csv', usecols=range(1, 15))
    print(train_data)
    train_data = np.array(train_data)
    x_train = train_data[:, 1:]
    y_train = train_data[:, 0].reshape(-1, 1)

    test_data = pd.read_csv('./data/test.csv', usecols=range(0, 14))
    test_data = np.array(test_data)
    test_id = test_data[:, 0]
    test_data = test_data[:, 1:]

    return x_train, y_train, test_id, test_data


def data_interpolation(train_data, test_data):
    """
    利用每一列特征的均值替换原始数据中存在的nan
    :param train_data: 可能包含nan的训练集
    :param test_data: 可能包含nan的测试集
    :return: 均值替换nan后的数据
    """
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0, copy=True, verbose=0)
    imp.fit(train_data)
    train_data_inter = imp.transform(train_data)
    test_data_inter = imp.transform(test_data)
    return train_data_inter, test_data_inter


def data_process(data):
    """
    数据标准化
    :param data: 输入的不含nan的数据
    :return: 处理后的数据
    """
    # min_max_scaler = MinMaxScaler()
    # data_mixmax = min_max_scaler.fit_transform(data)  # 最小最大值标准化，将数据压缩至（0,1）范围
    data_NL = scale(data, axis=0) # 去均值，方差标准化为1
    # data_NR = normalize(data, norm='l2')  # L2正则化
    return data_NL




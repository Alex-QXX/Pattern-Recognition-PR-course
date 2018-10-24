# -*- coding: utf-8 -*-
"""
Created on Oct 21 2018
@author:  秦祥翔
"""

from sklearn.preprocessing import StandardScaler
import numpy as np

def NL(data):
    Scaler = StandardScaler()
    Scaler.fit(data)
    Data = Scaler.transform(data)
    NewData = np.mat(Data)
    return NewData
    
def zeromean(data):
    meandata = np.mean(data, axis=0)
    NewData = data - meandata
    return NewData, meandata    
    
def datapca(data, k):

  data_NL = NL(data)

  data_NL_mean, datamean = zeromean(data_NL)  # 去均值和方差归一化

  datacov = np.mat(np.cov(data_NL_mean, rowvar=0))  # 协方差矩阵

  Lambda, P = np.linalg.eig(datacov)  # 协方差矩阵的特征值和特征向量

  Lambda_reorder_index = np.argsort(-Lambda)  # 特征值从大到小排列并得到索引

  print("降至 %d 维" % k)
  n_index = Lambda_reorder_index[:k]
  P_n = P[:, n_index]  # 根据降的维度取协方差矩阵特征向量的前n个
  data_n = np.dot(data_NL_mean, P_n)  # 获得降维数据
  recondata = np.dot(data_n, P_n.T) + datamean
  return data_n, recondata

# -*- coding: utf-8 -*-
"""
Created on Sept 23 2018
@author:  ALEX
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加字体使得画图显示中文

X = np.array([6.19, 2.51, 7.29, 7.01, 5.7, 2.66, 3.98, 2.5, 9.1, 4.2])
Y = np.array([5.25, 2.83, 6.41, 6.71, 5.1, 4.23, 5.05, 1.98, 10.5, 6.3])
# X = np.array([8.19, 2.72, 6.39, 8.71, 4.7, 2.66, 3.78])
# Y = np.array([7.01, 2.78, 6.47, 6.71, 4.1, 4.23, 4.05])
# 绘制散点
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, color="green", label="样本数据", linewidth=2)

X = X.reshape((-1, 1))
row, _ = X.shape
X = np.hstack((np.ones([row, 1]), X))  # 添加一列全1
# Y = Y.reshape((-1, 1))

row, col = X.shape
# b = [[0], [0]]
b = np.random.random([col, 1])
alpha = 0.01  # 学习率
delta_b = 1/row * (X.T.dot(Y.reshape(row, 1)) - X.T.dot(X).dot(b))
new_b = b + alpha * delta_b

# 梯度下降求loss的极小值
loss = []
for i in range(100000):
    tmp_loss = 1 / row * np.linalg.norm(Y.reshape(row, 1) - X.dot(b)) ** 2
    loss.append(tmp_loss)
    if tmp_loss < 0.01:
        print(i)
        break
    else:  # 更新权重
        b = new_b
        delta_b = 1 / row * (X.T.dot(Y.reshape(row,1)) - X.T.dot(X).dot(b))
        new_b = b + alpha * delta_b
print(b)
x = np.linspace(0, 12, 100)  # 在0-12直接画100个连续点
y = b[1]*x+b[0]  # 函数式

plt.plot(x, y, color="red", label="拟合直线", linewidth=2)
plt.legend(loc='lower right')  # 绘制图例
plt.show()
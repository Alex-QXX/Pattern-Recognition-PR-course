import numpy as np
from tkinter import _flatten
import pandas as pd
prediction = np.load('./prediction.npy')
prediction = prediction.reshape(400, 100)

f = open("./data/ss100_test.txt", "r")   # 读取test数据
data = f.readlines()  # 直接将文件中按行读到list里
f.close()             # 关闭文件
x_test = []
for index, seq in enumerate(data):
    if index % 2 == 1:
        x_test.append(seq)

# x_test = np.array(x_test)
re_predict = []
for index, one_seq in enumerate(x_test):
    one_seq = one_seq[:-1]
    seq_len = len(one_seq)  # 根据每条氨基酸链的长度将裁剪预测类别数据
    pre_list = list(prediction[index])
    del pre_list[seq_len:100]  # 删除不存在的单体对应的类别
    re_predict.append(pre_list)

re_predict = list(_flatten(re_predict))
# 存储预测数据
data = pd.DataFrame({'class': re_predict})
data.to_csv('submit_2.csv', index=False)

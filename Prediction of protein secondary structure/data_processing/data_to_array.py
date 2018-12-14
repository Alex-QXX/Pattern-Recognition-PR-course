import numpy as np
from sklearn.model_selection import train_test_split

def train_to_array(data_x, data_y):
    X = []
    Y = []
    for one_sequence in data_x:
        one_sample = []
        one_sequence = one_sequence[:-1]
        print(one_sequence)
        for feature in one_sequence:
            rows = np.zeros(21)
            if feature == 'A':  # 根据对应的字母确定该行的某一列为1
                rows[0] = 1
            elif feature == 'R':
                rows[1] = 1
            elif feature == 'N':
                rows[2] = 1
            elif feature == 'D':
                rows[3] = 1
            elif feature == 'C':
                rows[4] = 1
            elif feature == 'Q':
                rows[5] = 1
            elif feature == 'E':
                rows[6] = 1
            elif feature == 'G':
                rows[7] = 1
            elif feature == 'H':
                rows[8] = 1
            elif feature == 'I':
                rows[9] = 1
            elif feature == 'L':
                rows[10] = 1
            elif feature == 'K':
                rows[11] = 1
            elif feature == 'M':
                rows[12] = 1
            elif feature == 'F':
                rows[13] = 1
            elif feature == 'P':
                rows[14] = 1
            elif feature == 'S':
                rows[15] = 1
            elif feature == 'T':
                rows[16] = 1
            elif feature == 'W':
                rows[17] = 1
            elif feature == 'Y':
                rows[18] = 1
            elif feature == 'V':
                rows[19] = 1
            elif feature == 'X':
                rows[20] = 1
            else:
                break
            one_sample.append(rows)

        one_sample = np.array(one_sample)
        print(one_sample.shape)
        non_seq = np.zeros(22)
        non_seq[-1] = 1
        non_seq.shape = (1, 22)
        # 需要填充长度小于100的序列，填充的每一行的最后一列为1
        for i in range(100-one_sample.shape[0]):
            one_sample = np.r_[one_sample, non_seq]
        X.append(one_sample)
    X = np.array(X)
    print(X.shape)

    for one_sequence in data_y:
        one_sample = []
        one_sequence = one_sequence[:-1]
        print(one_sequence)
        for label in one_sequence:
            rows = np.zeros(4)
            if label == 'H':
                rows[0] = 1
            elif label == 'E':
                rows[1] = 1
            elif label == 'C':
                rows[2] = 1
            else:
                break
            one_sample.append(rows)
        one_sample = np.array(one_sample)
        print(one_sample.shape)
        non_seq = np.zeros(4)
        non_seq[-1] = 1
        non_seq.shape = (1, 4)
        for i in range(100 - one_sample.shape[0]):
            one_sample = np.r_[one_sample, non_seq]
        Y.append(one_sample)
    Y = np.array(Y)
    print(Y.shape)
    return X, Y


def test_to_array(data_x):
    X=[]
    for one_sequence in data_x:
        one_sample = []
        one_sequence = one_sequence[:-1]
        print(one_sequence)
        for feature in one_sequence:
            rows = np.zeros(21)
            if feature == 'A':
                rows[0] = 1
            elif feature == 'R':
                rows[1] = 1
            elif feature == 'N':
                rows[2] = 1
            elif feature == 'D':
                rows[3] = 1
            elif feature == 'C':
                rows[4] = 1
            elif feature == 'Q':
                rows[5] = 1
            elif feature == 'E':
                rows[6] = 1
            elif feature == 'G':
                rows[7] = 1
            elif feature == 'H':
                rows[8] = 1
            elif feature == 'I':
                rows[9] = 1
            elif feature == 'L':
                rows[10] = 1
            elif feature == 'K':
                rows[11] = 1
            elif feature == 'M':
                rows[12] = 1
            elif feature == 'F':
                rows[13] = 1
            elif feature == 'P':
                rows[14] = 1
            elif feature == 'S':
                rows[15] = 1
            elif feature == 'T':
                rows[16] = 1
            elif feature == 'W':
                rows[17] = 1
            elif feature == 'Y':
                rows[18] = 1
            elif feature == 'V':
                rows[19] = 1
            elif feature == 'X':
                rows[20] = 1
            else:
                break
            one_sample.append(rows)

        one_sample = np.array(one_sample)
        print(one_sample.shape)
        non_seq = np.zeros(22)
        non_seq[-1] = 1
        non_seq.shape = (1, 22)
        for i in range(100-one_sample.shape[0]):
            one_sample = np.r_[one_sample, non_seq]
        X.append(one_sample)
    X = np.array(X)
    print(X.shape)
    return X


f = open("./data/ss100_train.txt", "r")   #设置文件对象
data = f.readlines()  #直接将文件中按行读到list里
f.close()             #关闭文件

x_train = []
y_train = []
x_y = []
for index, seq in enumerate(data):
    if index % 2 == 1:
        x_y.append(seq)  # 将偶数行取出来

for index, train_data in enumerate(x_y):
    if index % 2 == 0:
        x_train.append(train_data)  # 每个氨基酸序列
    elif index % 2 == 1:
        y_train.append(train_data)  # 对应的标签

x_data, y_data = train_to_array(x_train, y_train)  # 将数据处理成二维数组
train_X, validation_X, train_y, validation_y = train_test_split(x_data, y_data, test_size=0.25)
np.save('./train_X.npy', train_X)
np.save('./train_y.npy', train_y)
np.save('./val_X.npy', validation_X)
np.save('./val_y.npy', validation_y)

f = open("./data/ss100_test.txt", "r")   #设置文件对象
data = f.readlines()  #直接将文件中按行读到list里
f.close()             #关闭文件
x_test = []
for index, seq in enumerate(data):
    if index % 2 == 1:
        x_test.append(seq)
# print(x_test)
test_X = test_to_array(x_test)
# print(test_X)
np.save('./test_X.npy',test_X)

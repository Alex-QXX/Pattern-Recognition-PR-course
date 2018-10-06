import numpy as np

# 输入 input
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# 输出 output
y = np.array([[0],
              [1],
              [1],
              [0]])

# 定义权重，随机生成各层每个节点之间的连接权重
np.random.seed(1)  # 设置随机数种子
# 输入层3个节点，隐层4个节点，输出层1个节点
W0 = 2 * np.random.random((3, 4)) - 1  # 随机生成3行4列的矩阵
W1 = 2 * np.random.random((4, 1)) - 1  # 乘以2再减1 矩阵每个元素∈（-1,1）


# 定义非线性方程
def sigmoid(X, derive=False):
    if not derive:
        return 1 / (1 + np.exp(-X))
    else:
        return sigmoid(X) * (1 - sigmoid(X))


# 进行训练，学习
training_times = 60000  # 训练次数
for i in range(training_times):
    # 对每一层输入输出进行计算
    # 输入层-隐层
    A0 = np.dot(X, W0)
    Z0 = sigmoid(A0)

    # 隐层-输出层
    A1 = np.dot(Z0, W1)
    _y = Z1 = sigmoid(A1)

    # 损失函数为 cost = ((y - _y)**2)/2，因使用梯度下降，故求导后得：
    cost = _y - y
    print("cost:{}".format(np.mean(np.abs(cost))))
    # 梯度下降使损失函数取最小
    delta_A1 = cost * sigmoid(Z1, derive=True)  # 对应笔记中的gj，j表示的是输出层第j个神经元,该例子中得到的是4*1的矩阵
    delta_W1 = np.dot(Z0.T, delta_A1)  # △w = gj * bh，bh为隐层第h个神经元的输出，Z0是4*4的矩阵，所以得到的是4*1的矩阵

    delta_A0 = np.dot(delta_A1, W1.T) * Z0 * (1 - Z0)
    # 对应笔记中的eh, eh = bh*(1-bh)* ∑ whj * gj,因为只有一个输出神经元，所以j=1 得到的是4*4的矩阵
    delta_W0 = np.dot(X.T, delta_A0)  # △γ = eh * x x是4*3的矩阵，所以得到3*4的矩阵

    # 更新参数
    rate = 0.1  # 学习率 ∈（0,1）通常为0.1
    W1 -= rate * delta_W1
    W0 -= rate * delta_W0
else:
    print('output:')
    print(_y)

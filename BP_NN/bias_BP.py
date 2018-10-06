import numpy as np

np.random.seed(1)

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

# 定义权重和偏置（也可以说是阈值）
W1 = 2 * np.random.random((3, 4)) - 1
W2 = 2 * np.random.random((4, 1)) - 1
b1 = 0.1 * np.ones((4,))  # 隐层各节点的阈值
b2 = 0.1 * np.ones((1,))  # 输出层节点的阈值


#  定义非线性方程
def sigmoid(X, derive=False):
    if not derive:
        return 1 / (1 + np.exp(-X))
    else:
        return sigmoid(X)*(1-sigmoid(X))


# 层数较多时，效果优于Sigmoid
def relu(X, derive=False):
    if not derive:
        return np.maximum(0, X)
    else:
        return (X > 0).astype(float)

no_line_fun = relu
# 进行训练，学习
training_times = 60000  # 训练次数
for time in range(training_times):
    A1 = np.dot(X, W1) + b1
    Z1 = no_line_fun(A1)

    A2 = np.dot(Z1, W2) + b2
    _y = Z2 = no_line_fun(A2)

    cost = _y - y
    print("error {}".format(np.mean(np.abs(cost))))

    delta_A2 = cost * no_line_fun(Z2, derive=True)
    delta_b2 = delta_A2.sum(axis=0)
    delta_W2 = np.dot(Z1.T, delta_A2)

    delta_A1 = np.dot(delta_A2, W2.T) * no_line_fun(Z1, derive=True)
    delta_b1 = delta_A1.sum(axis=0)
    delta_W1 = np.dot(X.T, delta_A1)

    rate = 0.1
    W1 -= rate * delta_W1
    b1 -= rate * delta_b1
    W2 -= rate * delta_W2
    b2 -= rate * delta_b2
else:
    print("output:")
    print(_y)

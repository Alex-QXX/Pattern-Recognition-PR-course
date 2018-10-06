import numpy as np


# 给出相关函数的定义
def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1.0 - np.tanh(x)*np.tanh(x)


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))


class NeuralNetwork:  # 定义神经网络类
    def __init__(self, layers, activation='tanh'):
        """
        :param layers: 包含每一层的单元数
        :param activation: 选择非线性函数是sigmoid还是tanh
        并初始化权重
        """
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_deriv = sigmoid_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        self.weights = []  # 定义权重
        self.weights.append((2 * np.random.random((layers[0] + 1, layers[1])) - 1) * 0.25)
        # 这里对于第一层的单元数加一是因为存在一个偏置量
        for i in range(2, len(layers)):  # 初始化连接权重
            self.weights.append((2 * np.random.random((layers[i - 1], layers[i])) - 1) * 0.25)
            # 产生随机矩阵并且去中心化，且控制在-0.25与+0.25之间
            # self.weights.append((2*np.random.random((layers[i]+1, layers[i+1]))-1)*0.25)

    def __str__(self):
        return str(self.weights)

    # 算法实现，主要是对每个参数进行不断更新,即学习过程
    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        X = np.atleast_2d(X)  # 确保
        temp = np.ones([X.shape[0], X.shape[1] + 1])
        temp[:, 0:-1] = X
        X = temp
        y = np.array(y)

        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l], self.weights[l])))

            error = y[i] - a[-1]
            deltas = [error * self.activation_deriv(a[-1])]

            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[1]))
            deltas.reverse()

            for i in range(len(self.weights)):
                layers = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layers.T.dot(delta)

    # 实现预测
    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return(a)


if __name__ == '__main__':
    # 给出一组数据进行预测
    nn = NeuralNetwork([2, 2, 1], 'sigmoid')
    # print(nn)
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([1, 0, 0, 1])
    nn.fit(x, y, 0.1, 10000)
    for i in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        print(i, nn.predict(i))
    print(nn)

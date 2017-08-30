import numpy as np


def tanh(x):
    """双曲正弦函数"""
    return np.tanh(x)


def tanh_deriv(x):
    """双曲正弦函数的导数"""
    return 1.0 - np.tanh(x)**2


def logistic(x):
    """逻辑函数"""
    return 1/(1+np.exp(-x))


def logistic_deriv(x):
    """逻辑函数的导数"""
    return logistic(x)*(1-logistic(x))


class NeuralNetwork(object):
    """定义一个神经网路类"""
    def __init__(self, layers, activation='tanh'):
        """初始化"""
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_deriv
        else:
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        self.weights = list()
        # layers = [10,2,2] 输入层（单层神经元个数），隐藏层，输出层
        # 权重和偏置的初始化
        for i in range(1, len(layers)-1):
            self.weights.append((2*np.random.random((layers[i-1]+1, layers[i]+1))-1)*0.25)  # +1表示偏向bais
            self.weights.append((2*np.random.random((layers[i]+1, layers[i+1]))-1)*0.25)

    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        """训练更新权重"""
        X = np.atleast_2d(X)
        # 添加偏向到输入层
        temp = np.ones((X.shape[0], X.shape[1]+1))
        temp[:, 0:-1] = X  # 除了最后一列
        X = temp
        y = np.array(y)

        for k in range(epochs):
            i = np.random.randint(X.shape[0])  # 随机选取一个数据
            O_j = [X[i]]
            # 神经网络每一层正向更新
            for j in range(len(self.weights)):
                O_j.append(self.activation(np.dot(O_j[j], self.weights[j])))  # 计算每个节点值I_j对应O_j
            # 反向传播开始
            #  计算最底层的误差
            error = y[i] - O_j[-1]  # (真实值标签T_j - 计算的标签O_j)
            deltas = [error * self.activation_deriv(O_j[-1])]  # (T_j - O_j)*func'() 最后层更新后的误差

            # 计算更新误差（从倒数2层到输入层）
            for l in range(len(O_j)-2, 0, -1):  # -2表示输出层和输入层不需要计算
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(O_j[l]))
            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(O_j[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        """预测"""
        x = np.array(x)
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        a_pred = temp
        for l in range(0, len(self.weights)):
            a_pred = self.activation(np.dot(a_pred, self.weights[l]))
        return a_pred




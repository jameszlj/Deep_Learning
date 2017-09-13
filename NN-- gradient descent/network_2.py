import numpy as np
import random
import json
import sys


class QuandrticCross(object):
    """二次cross"""
    @staticmethod
    def func(a, y):
        """激活输出a和期望输出y"""
        return 0.5 * np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """输出层误差"""
        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):
    """cross-entropy"""
    @staticmethod
    def func(a, y):
        """激活输出a和期望输出y"""
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """输出层误差"""
        return (a-y)


class Network(object):
    """BP神经网络类"""
    def __init__(self, sizes, cost=CrossEntropyCost):
        """定义初始化权重方法和神经网络参数"""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.cost = cost   # 选择cost函数
        self.default_weight_initializer()

    def feed_forward(self, a):
        """Return the output of  the network if a is input """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def default_weight_initializer(self):
        """默认使用均值为0，标准差为（1/sqrt(n_in)）初始化权重"""
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                       for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """使用标准正态分布初始化权重"""
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def SGD(self, training_data, epochs, mini_batch_size, eta,
        lmbda=0.0,
        evaluation_data=None,
        monitor_evaluation_cost=False,
        monitor_evaluation_accuracy=False,
        monitor_training_cost=False,
        monitor_training_accuracy=False):
        """随机梯度算法"""
        if evaluation_data:
            n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            # 每轮取一小批数据获得权重和偏向更新
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, n)
            print("Epoch %s training complete" % j)
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(
                    self.accuracy(evaluation_data), n_data))
            print()
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, min_batch, eta, lmbda, n):
        """反向传播更新权重和偏向"""
        # 初始化nabla_b 和nabla_w 的向量维度，并保存
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # 遍历mini_batch, 获取并分别更新到列表里
        for x, y in min_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(min_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(min_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """返回cost函数对w,b 的偏导数"""
        # 初始化nabla_b 和nabla_w 的向量维度，并保存
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 向前更新神经网络
        activation = x  # 初始化输入层
        activations = [x]  # list储存所有层的activation
        zs = list()  # 存储所有的z 向量
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # 反向更新
        # 输出层
        delta = self.cost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        # 隐藏层
        for layer in range(2, self.num_layers):
            # 从输出层的前一层往前推向输入层
            z = zs[-layer]
            delta = np.dot(self.weights[-layer+1].T, delta) * sigmoid_prime(z)
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta, activations[-layer-1].T)
        return (nabla_b, nabla_w)

    def total_cost(self, data, lmbda, convert=False):
        """返回总的正则化cost函数"""
        cost = 0.0
        for x, y in data:
            a = self.feed_forward(x)
            if convert:
                y = vectorized_result(y)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def accuracy(self, data, convert=False):
        """计算准确率总数"""
        if convert:
            results = [(np.argmax(self.feed_forward(x)), np.argmax(y))
                       for x, y in data]
        else:
            results = [(np.argmax(self.feed_forward(x)), y)
                       for x, y in data]
        return sum(int(x == y) for (x, y) in results)

    def save(self, filename):
        """保存神经网络参数到文件里"""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, 'w')
        json.dump(data, f)
        f.close()


def load(filename):
    """导入json文件"""
    f = open(filename, 'r')
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biasse = [np.array(b) for b in data["biases"]]
    return net


def sigmoid(z):
    """激活函数"""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """激活函数的导数"""
    return sigmoid(z)*(1-sigmoid(z))


def vectorized_result(j):
    """向量化结果"""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e











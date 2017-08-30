import numpy as np
import random
#import mnist_loader


def sigmoid(z):
    """逻辑函数"""
    return 1/(1+np.exp(-z))


def sigmoid_deriv(z):
    """逻辑函数的导数"""
    return sigmoid(z)*(1-sigmoid(z))


class Network(object):
    """定义梯度下降类"""
    def __init__(self, sizes):
        """sizes: 每层神经元的个数 [2,3,1]"""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feed_forward(self, a):
        """Return the output of  the network if a is input """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, train_data, epochs, mini_batch_size, eta, test_data=None):
        """随机梯度算法"""
        # eta：学习率
        if test_data:
            n_test = len(test_data)
        n = len(train_data)
        for j in range(epochs):
            random.shuffle(train_data)  # 打乱数据集
            mini_batchs = [train_data[k:k+mini_batch_size]
                           for k in range(0, n, mini_batch_size)]
            for min_batch in mini_batchs:
                self.update_mini_batch(min_batch, eta)
            if test_data:
                # format 格式化输出
                print("Epoch {0}:{1}/{2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """反向传播更新权重和偏向"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_w)]

    def backprop(self, x, y):
        """返回cost函数对w,b 的偏导数"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # 向前更新神经网络
        activation = x  # 初始化输入层
        activations = [x]  # list储存所有层的activation
        zs = list()  # 存储所有的z 向量


        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """计算正确率"""
        test_results = [(np.argmax(self.feed_forward(x)), y)
                        for (x, y) in test_data]  # argmax()返回时最大值索引, test_data中y也是返回标签索引
        return sum(int(x == y) for (x, y) in test_results)  # 返回(x, y)相等个数

#train_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = Network([2,3,1])
print(net.biases)
print(net.weights)
print(net.feed_forward(np.array([[1,3,3],[2,4,6]])))

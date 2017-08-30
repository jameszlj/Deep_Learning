import numpy as np
import random


def get_data(num_point, bias, variance):
    """获取数据"""
    x = np.zeros((num_point, 2))
    y = np.zeros(num_point)
    for i in range(num_point):
        x[i][0] = 1
        x[i][1] = i
        y[i] = (i+bias)+random.uniform(0, 1) * variance
    return x, y


def gradient_descent(x, y, theta, alpha, m, num_iters):
    """梯度下降法"""
    x_trains = x.T
    for i in range(num_iters):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        cost = np.sum(loss ** 2)/(2 * m)
        print('Iteration (%d) cost:%f' % (i, cost))
        gradient = np.dot(x_trains, loss)/m
        theta = theta - alpha * gradient
    return theta

x, y = get_data(100, 25, 10)
print(x)
print(y)

m, n = np.shape(x)
num_iters = 100000
alpha = 0.0005
theta = np.ones(n)
theta = gradient_descent(x, y, theta, alpha, m, num_iters)
print(theta)

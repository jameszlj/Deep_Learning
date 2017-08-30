import numpy as np


def fit_SLR(x, y):
    """简单线性拟合"""
    n = len(x)
    dinominator = 0  # 分母
    numerator = 0  # 分子
    for i in range(0, n):
        numerator += (x[i] - np.mean(x))*(y[i] - np.mean(y))
        dinominator += (x[i]-np.mean(x))**2
    b1 = numerator/float(dinominator)
    b0 = np.mean(y)-b1*np.mean(x)

    return b0, b1


def predict(x, b0, b1):
    """预测"""
    return b0+x*b1

x = [1, 3, 2, 1, 3]
y = [14, 24, 18, 17, 27]
b0, b1 = fit_SLR(x, y)
y_predict = predict(6, b0, b1)
print('y_predic:' + str(y_predict))
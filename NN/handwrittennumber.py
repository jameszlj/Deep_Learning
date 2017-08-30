import numpy as np
from sklearn.datasets import load_digits
from NeuralNetwork import NeuralNetwork
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import  confusion_matrix, classification_report

digits = load_digits()
X = digits.data
y = digits.target
# 输入数据归一化
X -= X.min()
X /= X.max()

nn = NeuralNetwork([64, 100, 10], 'logistic')
X_train, X_test, y_train, y_test = train_test_split(X, y)  # 交叉验证法
labels_train = LabelBinarizer().fit_transform(y_train)  # sklearn 要求格式
labels_test = LabelBinarizer().fit_transform(y_test)

# 开始拟合：
nn.fit(X_train, labels_train, epochs=3000)

predictions = list()
for i in range(X_test.shape[0]):
    O_put = nn.predict(X_test[i])
    predictions.append(np.argmax(O_put))
print(confusion_matrix(y_test, predictions))  # 分类混淆矩阵
print(classification_report(y_test, predictions))  # 分类报告




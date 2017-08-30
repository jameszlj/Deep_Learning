from numpy import genfromtxt,array
from sklearn import linear_model

dataPath = r'Delivery.csv'
deliveryData = genfromtxt(dataPath, delimiter=',')  # csv以，间隔
print('data:%s' % deliveryData)
x = deliveryData[:, :-1]
y = deliveryData[:, -1]
print(x)
print(y)

lr = linear_model.LinearRegression()
lr.fit(x, y)
print(lr)
print(lr.coef_)  # 模型参数
print(lr.intercept_)  # 截距
x_Predict = [102, 6]
y_Predict = lr.predict(x_Predict)

print(y_Predict)

from numpy import genfromtxt
from sklearn import linear_model

dataPath = r'Delivery_Dummy.csv'
data = genfromtxt(dataPath, delimiter=',')

x = data[1:, :-1]
y = data[1:, -1]
mrl = linear_model.LinearRegression()
mrl.fit(x, y)

print(mrl.coef_)
print(mrl.intercept_)

x_predict = [90., 2., 0., 0., 1.]
y_predict = mrl.predict(x_predict)
print('--1--')
print(y_predict)

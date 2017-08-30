from sklearn import neighbors
from sklearn import datasets
# iris 一种花
knn = neighbors.KNeighborsClassifier()
iris = datasets.load_iris()

# 保存数据
# f = open('iris.data.csv', 'w')
# f.write(str(iris))
# f.close()
print(iris)

knn.fit(iris.data, iris.target)

predicted_label = knn.predict([[0.1, 0.2, 0.5, 0.4]])
print(predicted_label)


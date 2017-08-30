from sklearn import svm

x = [[2, 0], [1, 1], [2, 3]]
y = [0, 0, 1]
clf = svm.SVC(kernel='linear')
clf.fit(x, y)

print(clf)
# 获取支持向量点
print(clf.support_vectors_)
print('--1--')
# 获取支持向量点下标
print(clf.support_)
print('--2--')
# 支持向量点的个数
print(clf.n_support_)
print('--3--')
# 预测
print(clf.predict([2, 0]))

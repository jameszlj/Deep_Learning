from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
# print(__doc__)
# 控制台上显示日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# 导入数据
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
n_samples, h, w = lfw_people.images.shape  # shape 求图片矩阵的维度大小
# print(n_samples)  # t图片数量
# print(h, w)  # 图片的大小

X = lfw_people.data
n_features = X.shape[1]

y = lfw_people.target  # 提取不同人身份标记
target_names = lfw_people.target_names # 提取人的名字
n_classes = target_names.shape[0]

print('n_smaples:%d, n_features:%d, n_classes:%d' % (n_samples, n_features, n_classes))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# 特征提取/降维
n_components = 150
print('从%d个维度中提取到%d维度' % (X_train.shape[0], n_components))
t_start = time()
pca = PCA(n_components=n_components, whiten=True).fit(X_train)
print('done in %0.3fs' % (time()-t_start))

eigenfaces = pca.components_.reshape((n_components, h, w))
print('根据主成分降维开始')
t_start = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print(X_train_pca.shape[0])
print('done in %0.3fs' % (time()-t_start))

# 训练SVM
t_start = time()
# 构建归类精度 5*6
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print('done in %0.3fs' % (time()-t_start))
print(clf.best_estimator_)

# 测试集svm分类模型开始
# 预测测试集里面名字
t_start = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t_start))

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


# 画图
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# 画图预测部分测试集
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)


prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# 画人脸 eigenface 主成分特征脸

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()




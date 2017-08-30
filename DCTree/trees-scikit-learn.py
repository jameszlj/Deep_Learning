from sklearn.feature_extraction import DictVectorizer # 特征提取法法
import csv
from sklearn import tree
from sklearn import preprocessing # 数据预处理
from sklearn.externals.six import StringIO

# 读取csv文件
all_electronics_data = open('AllElectronics.csv', 'rU')  # rU 以读方式打开，同时提供通用换行符支持
reader = csv.reader(all_electronics_data)
hearders = next(reader)  # next 迭代

print(hearders)
# 特征值放入到list中
feature_list = list()
label_list = list()

for row in reader:
    label_list.append(row[len(row)-1])
    row_dict = dict()
    for i in range(1, len(row)-1):
        row_dict[hearders[i]] = row[i]
    feature_list.append(row_dict)

print(feature_list)

# 特征值实例化
vec = DictVectorizer()
# 特征值转化成整型数据矩阵如age(youth,middle,old)>>(1,0,0)
dummy_X = vec.fit_transform(feature_list).toarray()
print('dummy_X' + str(dummy_X))
print(vec.get_feature_names())
print('label_list:' + str(label_list))

# 标签类实例化
lb = preprocessing.LabelBinarizer()  # 将字符串类型的数据转化成整形
dummy_Y = lb.fit_transform(label_list)
print('label_list:' + str(dummy_Y))

# 使用决策树来分类
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(dummy_X, dummy_Y)
print('clf:' + str(clf))

# 可视化模型
# dot -Tpdf InformationGainOri.dot -o out_put.pdf 使用graphviz转化pdf
with open('InformationGainOri.dot', 'w') as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)

# 预测
one_row_x = dummy_X[0, :]
print('one_row_x:' + str(one_row_x))
new_row_x = one_row_x
new_row_x[0] = 1
new_row_x[2] = 0
print('new_row_x:' + str(one_row_x))
predicted_Y = clf.predict(new_row_x)
print('predicted_Y:' + str(predicted_Y))





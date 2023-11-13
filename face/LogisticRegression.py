from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn import preprocessing

# 读取数据集xlsx文件，利用pandas库根据列来分特征值和标签列
file_loc = "../数据集/数据.xlsx"
X = pd.read_excel(file_loc, index_col=None, na_values=['NA'], usecols="A:E")
Preprocessing = preprocessing.StandardScaler()
X = Preprocessing.fit_transform(X)
y = pd.read_excel(file_loc, index_col=None, na_values=['NA'], usecols="G")
print(X)
y = y.values.ravel()
# 调用Sklearn库中的SVM函数，并给分类器喂相应的数据
clf = LogisticRegression(multi_class='ovr', solver="liblinear")
acc = cross_val_score(clf, X, y, scoring='accuracy', cv=10).mean()
print("十倍交叉验证的准确率为", acc)
clf.fit(X, y)
# 可以根据前面介绍的参数，做出相应改变观察结果变化，采用ovr参数，实现多分
# 模型保存
joblib.dump(clf, "svm_model.pkl")
# 根据给定数据与标签返回正确率的均值
print(clf.score(X, y))

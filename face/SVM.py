from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import cross_val_score
import joblib


df = pd.read_excel("../数据集/train_dataset3.xlsx", header=None)
X = df[df.columns[0:6]]
y = df[df.columns[6]]
clf = SVC(kernel='poly')
acc = cross_val_score(clf, X, y, scoring='accuracy', cv=10).mean()
print("十倍交叉验证的准确率为", acc)
clf.fit(X, y)
acc = accuracy_score(y, clf.predict(X))

print("SVM模型分类准确率为", acc)
# 模型保存
joblib.dump(clf, "svm_model.pkl")

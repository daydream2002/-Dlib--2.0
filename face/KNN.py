import sklearn
import sklearn.neighbors as sk_neighbors
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

df = pd.read_excel("../数据集/数据.xlsx", header=None)
X = df[df.columns[0:5]]
y = df[df.columns[6]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 使用网格搜索优化参数
model = sk_neighbors.KNeighborsClassifier()
model = GridSearchCV(model, param_grid={'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19]}, cv=10)
acc = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=10).mean()
print("十倍交叉验证的准确率为", acc)
model.fit(X_train, y_train)
# 评估模型，根据给定数据与标签返回正确率的均值
acc = model.score(X_test, y_test)
print('KNN模型分类准确率:', acc)
# 保存模型
joblib.dump(model, 'knn_model.pkl')
print("Best parameters: ", model.best_params_)
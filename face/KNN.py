import sklearn
import sklearn.neighbors as sk_neighbors
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

df = pd.read_excel("../数据集/数据.xlsx", header=None)
X = df[df.columns[0:5]]
y = df[df.columns[6]]
#划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


model = sk_neighbors.KNeighborsClassifier()
#网格搜索
model = GridSearchCV(model, param_grid={'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19]}, cv=10)

model.fit(X_train, y_train)


print("在训练集上分类准确率为：", model.best_score_)
print("最好的超参数为 ", model.best_params_)
# 评估模型，根据给定数据与标签返回正确率的均值
acc = model.score(X_test, y_test)

print('在测试集上的分类准确率为:', acc)
# 保存模型
joblib.dump(model, 'knn_model.pkl')

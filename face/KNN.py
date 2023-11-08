import sklearn.neighbors as sk_neighbors
import joblib
import pandas as pd
from sklearn.model_selection import cross_val_score

# 读取数据集xlsx文件，利用pandas库根据列来分特征值和标签列
file_loc = "../数据集/数据.xlsx"
X = pd.read_excel(file_loc, index_col=None, na_values=['NA'], usecols="A:E")
y = pd.read_excel(file_loc, index_col=None, na_values=['NA'], usecols="G")
y = y.values.ravel()
# 给模型喂数据，调用Sklearn中的集成好的分类器函数
model = sk_neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=1)
acc = cross_val_score(model, X, y, scoring='accuracy', cv=10).mean()
print("十倍交叉验证的准确率为", acc)
model.fit(X, y)
# 评估模型，根据给定数据与标签返回正确率的均值
acc = model.score(X, y)
print('KNN模型分类准确率:', acc)
# 保存模型
joblib.dump(model, 'knn_model.pkl')

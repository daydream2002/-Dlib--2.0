import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pyswarms as ps
import pandas as pd

df = pd.read_excel("../数据集/数据.xlsx", header=None)
X = df[df.columns[0:5]]
y = df[df.columns[6]]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 定义SVM模型的参数优化目标函数
def optimize_svm(params):
    C= params[0]
    gamma = params[1]
    # 参数包括C（正则化参数）和gamma（核函数的系数）
    svm_model = SVC(C=C, gamma=gamma, kernel='rbf')  # 创建SVC模型
    svm_model.fit(X_train, y_train)  # 在训练集上拟合模型
    accuracy = svm_model.score(X_test, y_test)  # 在测试集上评估模型性能
    return 1 - accuracy  # 优化目标为最小化分类错误率


# 设置PSO算法的参数
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

# 定义参数空间
max_bound = np.array([100, 100])  # 参数上界
min_bound = np.array([0.1, 0.1])  # 参数下界
bounds = (min_bound, max_bound)

# 使用PSO算法进行参数优化
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options, bounds=bounds)
c = optimizer.optimize(optimize_svm, iters=100)

# 输出最佳参数组合
print("最佳参数组合：", c, g)

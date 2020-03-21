import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('./data/LogiReg_data.txt', names=['exam1','exam2','admitted'])
data.insert(0, 'Ones', 1)
orig_data = data.values
cols = orig_data.shape[1]
X = orig_data[:, 0:cols - 1]
y = orig_data[:, cols - 1:cols]
# 三个θ参数，用零占位
theta = np.zeros([1, 3])

#  显示数据图
def show_data(Data):
    fig = plt.figure(figsize=(10, 6))

    positive = Data[Data['admitted'] == 1]
    negative = Data[Data['admitted'] == 0]

    x_label_1 = positive['exam1']
    y_label_1 = positive['exam2']

    x_label_2 = negative['exam1']
    y_label_2 = negative['exam2']

    # plt.scatter(x_label_2, y_label_1)
    p1 = plt.scatter(x_label_1, y_label_1, s=80, c='g', marker='o')

    p2 = plt.scatter(x_label_2, y_label_2, s=80, c='r', marker='x')

    plt.xlabel('exam1')
    plt.ylabel('exam2')
    plt.legend([p1,p2],['admitted','not admitted'])

    plt.show()

# 构造sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def model(X, theta):
    return sigmoid(np.dot(X, theta.T)) # 矩阵乘法

def cost(X, y, theta):
    left = np.multiply(-y, np.log(model(X, theta)))
    right = np.multiply(1 - y, np.log(1 - model(X, theta)))
    return np.sum(left - right) / len(X)

def gradient(X, y, theta):
    grad = np.zeros(theta.shape) # 一共三个参数  所以计算三个参数的梯度
    print(grad)
    error = (model(X, theta) - y).ravel()# ravel展平数组
    print(error)
    for j in range(len(theta.ravel())):
        term = np.multiply(error, X[:, j])
        grad[0, j] = np.sum(term) / len(X) # grad[0, 0] grad[0, 1] grad[0, 2]
    return grad

gradient(X, y, theta)
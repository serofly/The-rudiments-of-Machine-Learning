from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report
import joblib
import pandas as pd
import numpy as np


def mylinear():
    """
    线性回归预测房价,分类是目标值离散，回归是目标值连续
    :return: nONE
    """
    # 获取数据
    lb = load_boston()
    # 分割
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)
    # 标准化， 回归当中目标值也需要标准化处理
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)
    std_y = StandardScaler()  # 目标值由于是一维的所以需要格式转化
    y_train = std_y.fit_transform(y_train.reshape(-1, 1))
    y_test = std_y.transform(y_test.reshape(-1, 1))
    # 算法
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    print("正规方程回归系数", lr.coef_)
    y_lr_predict = std_y.inverse_transform(lr.predict(x_test))  # 结果逆标准化
    print("测试集预测的房价", y_lr_predict)
    print("正规方程均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_lr_predict))
    # 保存训练好的模型
    joblib.dump(lr, "./lr_boston.pkl")
    # clf = joblib.load("./lr_boston.pkl"),加载模型，然后不需要训练，直接使用

    sgd = SGDRegressor()
    sgd.fit(x_train, y_train)
    print("梯度下降回归系数", sgd.coef_)
    y_sgd_predict = std_y.inverse_transform(sgd.predict(x_test))  # 结果逆标准化
    print("测试集预测的房价", y_sgd_predict)
    print("梯度下降均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_sgd_predict))

    rd = Ridge(alpha=1.0)  # alpha是超参数，可以取的值为0-1之间的小数，或者是1-10之间的整数
    rd.fit(x_train, y_train)  # 岭回归具有正则化，可以减少过拟合的现象
    print("岭回归系数", rd.coef_)
    y_rd_predict = std_y.inverse_transform(rd.predict(x_test))  # 结果逆标准化
    print("测试集预测的房价", y_rd_predict)
    print("岭回归均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_rd_predict))
    return 0


def logistic():
    """
    逻辑回归做二分类，自带正则处理过拟合问题，癌症预测
    逻辑回归的得出的概率指的是原本总数中小概率事件发生的概率
    :return: None
    """
    column = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniform of Cell Shape',
              'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli',
              'Mitoses', 'Class']
    data = pd.read_csv("./data/breast-cancer/breast-cancer-wisconsin.data", names=column)
    # print(data.head())

    # 缺失值处理，缺失值为？
    data = data.replace('?', np.nan)
    data = data.dropna()
    # 分割
    x = data[column[1:10]]
    y = data[column[10]]  # 2是良性，4是恶性
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 标准化
    std = StandardScaler()  # 二分类回归，所以目标值不需要标准化
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    # 算法
    lg = LogisticRegression(C=1.0)
    lg.fit(x_train, y_train)
    print("回归系数", lg.coef_)
    print("准确率：", lg.score(x_test, y_test))
    # 病症重点是恶性的召回率
    y_predict = lg.predict(x_test)
    print("召回率", classification_report(y_test, y_predict, labels=[2, 4], target_names=["良性", "恶性"]))

    return None


if __name__ == '__main__':
    logistic()

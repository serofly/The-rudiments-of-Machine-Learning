from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.model_selection import train_test_split


# 用于分类的小数据集
li = load_iris()
print(li.data)  # 特征值
print(li.target)  # 目标值

# 用于分类的大数据集
news = fetch_20newsgroups(subset="all")
print(news.data)
print(news.target)

# 导入特征值和目标值，以及测试集的占比
x_train, x_test, y_train, y_test = train_test_split(li.data, li.target, test_size=0.25)
print(x_train, y_train)
print(x_test, y_test)




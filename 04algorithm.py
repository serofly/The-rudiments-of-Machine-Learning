# K近邻算法，KNN
# 求未知样本与已知样本集中的最近距离，距离为欧式距离
# K近邻算法需要标准化处理
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier


def knncls():
    """
    K近邻预测用户签到位置
    优点：简单，无需估计参数，无需训练
    缺点：样本计算量大，必须指定K值
    使用场景：几千-几万样本量
    :return: None
    """
    data = pd.read_csv("./data/FBlocation/train.csv")
    # print(data.head())

    # 缩小数据
    data = data.query("x > 1.0 & x < 1.25 & y > 2.5 & y < 2.75")

    # 时间处理，将时间戳改为时间日期
    time_value = pd.to_datetime(data['time'], unit='s')
    # print(time_value)
    time_value = pd.DatetimeIndex(time_value)  # 时间日期变为字典格式

    # 增加特征：日、时、星期，其中年月都是一样的，而分秒等数据认为对签到位置无影响
    data = data.copy()  # 避免出现SettingWithCopyWarning，可以不用loc
    data['day'] = time_value.day
    data['hour'] = time_value.hour
    data['weekday'] = time_value.weekday

    # 删除时间戳特征的列
    data = data.drop(['time'], axis=1)
    # print(data.head())

    # 删除签到数量过少的地点
    place_count = data.groupby("place_id")['row_id'].count()  # 根据地点id分类，统计入住人数
    tf = place_count[place_count.values > 10]  # 删选签到人数大于10的地点，tf的格式时index是place_id，values是签到人数
    data = data[data['place_id'].isin(tf.index)]  # 这要这些大于10的地点的id
    # print(data)

    # 取出特征值和目标值
    y = data['place_id']
    x = data.drop(['place_id', 'row_id'], axis=1)

    # 进行数据分割,训练集为25%
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 特征工程,标准化
    std = StandardScaler()
    x_train = std.fit_transform(x_train)  # 只对特征值做标准化，目标值只是一列不需要标准化
    x_test = std.transform(x_test)  # 不需要再fit了，fit只是为了求平均值，方差等等用的，训练集和测试集来源一样，这些值也不会差太多

    # KNN算法
    knn = KNeighborsClassifier()  # 选择最近的n个
    # knn.fit(x_train, y_train)
    # y_predict = knn.predict(x_test)
    # print("预测值", y_predict)
    # print("准确l率", knn.score(x_test, y_test))  # 其实直接用时间戳，不分日、小时、星期结果更高

    # 网格搜素+交叉验证
    param = {"n_neighbors": [3, 5, 10]}
    gc = GridSearchCV(knn, param_grid=param, cv=5)  # 5折交叉验证
    gc.fit(x_train, y_train)
    print("测试集上的准确率：", gc.score(x_test, y_test))
    print("最优结果：", gc.best_score_)
    print("最好的模型：", gc.best_estimator_)
    print("每个超参数交叉验证的结果：", gc.cv_results_)
    return None


def naviebayes():
    """
    朴素贝叶斯进行文本分类
    优点：稳，准，快，简单，缺失数据不敏感，不用调参数
    缺点:样本要求高，样本属性之间需要独立性
    使用场景：常用于文本分类
    :return:None
    """
    news = fetch_20newsgroups(subset="all")
    # 数据分割
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25)

    # 对数据集进行特征抽取
    tf = TfidfVectorizer()
    x_train = tf.fit_transform(x_train)
    x_test = tf.transform(x_test)

    # 算法
    mlt = MultinomialNB(alpha=1.0)
    mlt.fit(x_train, y_train)
    y_predict = mlt.predict(x_test)
    print("预测值", y_predict)
    print("准确率：", mlt.score(x_test, y_test))
    print("精确率和召回率：", classification_report(y_test, y_predict, target_names=news.target_names))
    return None


def decison():
    """
    决策树预测titanic生存概率
    优点：可视化、不需要归一化
    缺点：过拟合
    优化: 随机森林
    :return: None
    """
    titanic = pd.read_csv("./data/titanic/train.csv")

    # 找到特征值和目标值
    x = titanic[['Age', 'Pclass', 'Sex', 'Embarked']]
    y = titanic['Survived']
    # 空缺值处理
    x = x.copy()  # 避免SettingWithCopyWarning
    x['Age'].fillna(x['Age'].mean(), inplace=True)
    x['Embarked'].fillna(x['Embarked'].mode()[0], inplace=True)

    # 数据集分割
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 特征工程
    dict = DictVectorizer(sparse=False)
    x_train = dict.fit_transform(x_train.to_dict(orient="records"))  # 按行进行转换成字典，每个字典再组成列表
    x_test = dict.transform(x_test.to_dict(orient="records"))
    print(dict.get_feature_names())

    # 决策树算法
    # dec = DecisionTreeClassifier(max_depth=5)
    # 默认是使用基尼系数，若乡选择最大信息增益的熵criterion="entropy";决策树分层数为max_depth,默认为train集100%预测率的分层数
    # dec.fit(x_train, y_train)
    # print("准确率：", dec.score(x_test, y_test))

    # 导出决策树结构
    # export_graphviz(dec, out_file="./titanic_tree.dot", feature_names=['年龄', 'C地登船', 'Q地登船', 'S地登船', '船舱等级', '女性', '男性'])
    # 记事本打开dot，在node属性中空格添加fontname="FangSong"，使其支持中文
    # cmd中输入dot -Tpng titanic_tree.dot -o titanic_tree.png转换格式

    # 随机森林算法
    rf = RandomForestClassifier()
    param = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}
    # 网格搜索与交叉验证
    gc = GridSearchCV(rf, param_grid=param, cv=3)
    gc.fit(x_train, y_train)
    print("准确率：", gc.score(x_test, y_test))
    print("最佳模型：", gc.best_params_)
    # 随机森林不能可视
    # 决策树和随机森林还有min_sample_split和min_sample_leaf两个参数，
    # min_sample_split小于这个数的节点将不会再开支分裂，
    # min_sample_leaf小于这个数的节点将被剪枝
    return None


if __name__ == '__main__':
    decison()

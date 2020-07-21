from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import jieba
import numpy as np


def dictvec():
    """
    字典数据抽取
    :return:None
    """
    dict = DictVectorizer(sparse=False)
    data = dict.fit_transform([{'city': '北京', 'temp': 100}, {'city': '上海', 'temp': 60}, {'city': '深圳', 'temp': 30}])
    print(dict.get_feature_names())  # 特征值
    print(data)  # one-hot编码值
    return None


def countvec():
    """
    文本特征值化
    :return:None
    """
    count = CountVectorizer()
    data = count.fit_transform(["Life is short, I like python", "Life is too long, I dislike python"])
    print(count.get_feature_names())
    print(data.toarray())  # 文本没有sparse=False，所以用toarray()。这里显示的是文本中词出现次数，不统计字母
    return None


def hanzivec():
    """
    汉字文本特征值化
    :return: None
    """
    c1 = cutword("如果你不能飞，那就奔跑；如果不能奔跑，那就行走；如果不能行走，那就爬行；但无论你做什么，都要保持前行的方向。")
    c2 = cutword("今天很残酷,明天更残酷,后天很美好,但绝大多数人死在明天晚上。")
    c3 = cutword("在历史的长河中，能够脱颖而出的那些人，并不是因为他们走得足够快，而是因为他们走得足够久。")
    count = CountVectorizer()
    text = count.fit_transform([c1, c2, c3])
    print(count.get_feature_names())
    print(text.toarray())
    return None


def cutword(text):
    """
    分词
    :return:c
    """
    con = jieba.cut(text)  # jieba生成一个分完词的迭代器
    c = " ".join(list(con))  # 需要的是带空格的字符串
    return c


def tfidfvec():
    """
    汉字文本重要性特征抽取
    :return: None
    """
    c1 = cutword("如果你不能飞，那就奔跑；如果不能奔跑，那就行走；如果不能行走，那就爬行；但无论你做什么，都要保持前行的方向。")
    c2 = cutword("今天很残酷,明天更残酷,后天很美好,但绝大多数人死在明天晚上。")
    c3 = cutword("在历史的长河中，能够脱颖而出的那些人，并不是因为他们走得足够快，而是因为他们走得足够久。")
    count = TfidfVectorizer()  # 词的重要性，在其他文章中出现过的，你这里的重要性程度就会降低
    text = count.fit_transform([c1, c2, c3])
    print(count.get_feature_names())
    print(text.toarray())
    return None


def mm():
    """
    归一化处理, 受异常点影响显示，目的是为了缩小不同特征加权时的差异
    :return:None
    """
    mm = MinMaxScaler(feature_range=(0, 1))
    data = mm.fit_transform([[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]])
    print(data)


def st():
    """
    标准化处理，需要大样本
    :return:None
    """
    st = StandardScaler()
    data = st.fit_transform([[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]])
    print(data)
    return None


def na():
    """
    空缺值填充
    :return: None
    """
    im = SimpleImputer(missing_values=np.nan, strategy="mean")
    data = im.fit_transform([[1, 2], [np.nan, 3], [7, 6]])
    print(data)
    return None


def var():
    """
    特征选择，删除低方差的特征，进行数据降维
    :return: None
    """
    var = VarianceThreshold(threshold=0.0)
    data = var.fit_transform([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])
    print(data)
    return None


def pca():
    """
    主成分分析，减少特征维度，但不损失太多信息，进行数据降维
    :return: None
    """
    pca = PCA(n_components=0.9)  # 数据累积贡献率不低于90%，损失不超过10%
    data = pca.fit_transform([[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]])
    print(data)
    return None


if __name__ == '__main__':
    pca()

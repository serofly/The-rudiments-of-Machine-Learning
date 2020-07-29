import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import numpy as np

# 中文和负号的正常显示
plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False


# 读取四张表数据
prior = pd.read_csv("./data/PCA/order_products__prior.csv")  # product_id, order_id
products = pd.read_csv("./data/PCA/products.csv")  # product_id,aisle_id
orders = pd.read_csv("./data/PCA/orders.csv")  # order_id, user_id
aisle = pd.read_csv("./data/PCA/aisles.csv")  # aisle_id, aisle

# 合并四张表，得到用户-物品类型
_mg = pd.merge(prior, products, left_on="product_id", right_on="product_id")
_mg = pd.merge(_mg, orders, left_on="order_id", right_on="order_id")
mt = pd.merge(_mg, aisle, left_on="aisle_id", right_on="aisle_id")

# 交叉表
cross = pd.crosstab(mt['user_id'], mt['aisle'])
# 主成分分析
pca = PCA(n_components=0.9)
data = pca.fit_transform(cross)
print(data.shape)
# 假设一共有4个类别的用户,维度不高可以不用主成分分析
data = data[:500]
km = KMeans(n_clusters=4)
km.fit(data)
predict = km.predict(data)  # 每个数变成了0，1，2，3
# km.labels_获取聚类标签//km.cluster_centers_ #获取聚类中心
print("聚类分析的轮廓系数", silhouette_score(data, predict))  # -1到0为差，0-1为好，越靠近1越好

# 显示聚类结果散点图
plt.figure(figsize=(10, 10))
color = ["orange", "green", "blue", "red"]
col = [color[i] for i in predict]
plt.scatter(data[:, 1], data[:, 20], color=col)
plt.xlabel("1")
plt.ylabel("20")
plt.show()

# 图例带人数的雷达图
plt.figure(figsize=(10, 8))
plt.subplot(111, polar=True)
plt.style.use('ggplot')
# 统计频数
r1 = pd.Series(km.labels_).value_counts()
# 将n个簇类中心转换成DataFrame格式
r2 = pd.DataFrame(km.cluster_centers_).iloc[:, :6]
center = pd.concat([r2, r1], axis=1)
feature = [1, 2, 3, 4, 5, 6]
angles = np.linspace(0, 2 * np.pi, len(feature), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))
for i, j in enumerate(center.values):  # i 是行，即第几类人；j是列，表示feature+r1
    # 为了使雷达图一圈封闭，需要以下步骤
    values = np.concatenate((j[:-1], [j[0]]))
    # 绘制折线图
    plt.plot(angles, values, 'o-', linewidth=2, label='第%d类人群，%d人' % (i + 1, j[-1]))
    # 填充颜色
    plt.fill(angles, values, col[i], alpha=0.25)
# 添加每个特质的标签
plt.thetagrids(angles * 180 / np.pi, feature)
# 添加标题
plt.title('顾客分类状况')
# 设置图例
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), ncol=1, fancybox=True, shadow=True)
# 显示图形
plt.show()

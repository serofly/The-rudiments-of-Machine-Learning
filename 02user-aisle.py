import pandas as pd
from sklearn.decomposition import PCA

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
print(data)

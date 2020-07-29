# 预测全部用户在12月的借款总额
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib


def load_csv(csv_name):
    """导入csv文件"""
    csv_file = pd.read_csv("./data/JDloan/{}.csv").format(csv_name)
    return csv_file


def get_month(data):
    """获得月份"""
    return int(data.split('-')[1])


def get_amount(data):
    """获得金额解密"""
    return round(5 ** data - 1)


def order_feature(t_order, t_user, MONTH):
    """MONTH为训练集的月份数"""
    # 空缺值填充,选择时间集。          8月-10月，11月||||8-11月，12月
    t_order.fillna(0, inplace=True)
    t_order["month"] = t_order["buy_time"].apply(get_month)
    t_order = t_order[t_order.month < (8 + MONTH)]

    # 金额解密
    t_order["price"] = t_order["price"].apply(get_amount)
    t_order["discount"] = t_order["discount"].apply(get_amount)

    # 数据price填充
    cate_mean_price = t_order[t_order.price != 0].groupby("cate_id")["price"].mean().rename(
        "cate_mean_price").reset_index()
    t_order = t_order.merge(cate_mean_price, on="cate_id", how="left")
    t_order.loc[t_order.price <= 0, "price"] = t_order.loc[t_order.price <= 0, "cate_mean_price"]

    # 每笔实际总消费
    t_order["real_price"] = t_order["price"] * t_order["qty"] - t_order["discount"]
    t_order.loc[t_order["real_price"] < 0, "real_price"] = 0

    # 物品的实际单价
    t_order["per_price"] = (t_order["real_price"] + t_order["discount"]) / t_order["qty"]

    # 折扣率
    t_order["discount_ratio"] = 1 - t_order["real_price"] / (t_order["price"] * t_order["qty"])

    # 每个人平均每笔实际购物总价，整合到user表中
    user_price_sum_mean = t_order.groupby(["uid"])["real_price"].mean().rename("price_sum_mean").reset_index()
    t_user = t_user.merge(user_price_sum_mean, on=["uid"], how="left")
    # 每个人平均每笔实际购物单价，整合到user表中
    user_price_mean = t_order.groupby(["uid"])["per_price"].mean().rename("price_mean").reset_index()
    t_user = t_user.merge(user_price_mean, on=["uid"], how="left")
    # 每个人折扣率平均值，整合到user表中
    user_discount_radio_mean = t_order.groupby(["uid"])["discount_ratio"].mean().rename(
        "discount_radio_mean").reset_index()
    t_user = t_user.merge(user_discount_radio_mean, on=["uid"], how="left")
    # 每个人平均每月购物笔数，整合到user表中
    user_buy_count_month = t_order.groupby(["uid"])["uid"].count().rename("buy_count_month").reset_index()
    user_buy_count_month['buy_count_month'] = user_buy_count_month['buy_count_month'] / MONTH
    t_user = t_user.merge(user_buy_count_month, on=["uid"], how="left")
    # 每个人平均每月实际购物总价，整合到user表中
    user_price_sum_month = t_order.groupby(["uid"])["real_price"].sum().rename("price_sum_month").reset_index()
    user_price_sum_month['price_sum_month'] = user_price_sum_month['price_sum_month'] / MONTH
    t_user = t_user.merge(user_price_sum_month, on=["uid"], how="left")
    # 每个人平均每月实际购物总折扣金额，整合到user表中
    user_discount_sum_month = t_order.groupby(["uid"])["discount"].sum().rename("discount_sum_month").reset_index()
    user_discount_sum_month['discount_sum_month'] = user_discount_sum_month['discount_sum_month'] / MONTH
    t_user = t_user.merge(user_discount_sum_month, on=["uid"], how="left")
    t_user.fillna(0)
    return t_user


def loan_feature(t_loan, t_user, MONTH):
    t_loan["month"] = t_loan["loan_time"].apply(get_month)
    t_loan = t_loan[t_loan.month < (8 + MONTH)]

    # 金额解密
    t_loan["loan_amount"] = t_loan["loan_amount"].apply(get_amount)
    # 每笔每期需还款金额
    t_loan["ave_loan_amount"] = t_loan["loan_amount"] / t_loan["plannum"]

    # 每个人平均每月借款总额，整合到user表中
    user_loan_sum = t_loan.groupby(["uid"])["loan_amount"].sum().rename("loan_sum_month").reset_index()
    user_loan_sum["loan_sum_month"] = user_loan_sum["loan_sum_month"] / MONTH
    t_user = t_user.merge(user_loan_sum, on=["uid"], how="left")
    # 每个人平均每月借款笔数，整合到user表中
    user_loan_count = t_loan.groupby(["uid"])["uid"].count().rename("loan_count_month").reset_index()
    user_loan_count["loan_count_month"] = user_loan_count["loan_count_month"] / MONTH
    t_user = t_user.merge(user_loan_count, on=["uid"], how="left")
    # 每个人平均每月还款总额，整合到user表中
    user_ave_loan_sum = t_loan.groupby(["uid"])["ave_loan_amount"].sum().rename("ave_loan_sum").reset_index()
    user_ave_loan_sum["ave_loan_sum"] = user_ave_loan_sum["ave_loan_sum"] / MONTH
    t_user = t_user.merge(user_ave_loan_sum, on=["uid"], how="left")
    # 每个人平均每笔借款总额，整合到user表中
    user_loan_mean = t_loan.groupby(["uid"])["loan_amount"].mean().rename("loan_mean").reset_index()
    t_user = t_user.merge(user_loan_mean, on=["uid"], how="left")
    # 每个人平均每笔借款还款数，整合到user表中
    user_plannum_mean = t_loan.groupby(["uid"])["plannum"].mean().rename("plannum_mean").reset_index()
    t_user = t_user.merge(user_plannum_mean, on=["uid"], how="left")
    t_user.fillna(0)
    return t_user


def click_feature(t_click, t_user, MONTH):
    t_click["month"] = t_click["click_time"].apply(get_month)
    t_click = t_click[t_click.month < (8 + MONTH)]
    # 每个人平均每月点击页面数，整合到user表中
    user_click_count = t_click.groupby(["uid"])["uid"].count().rename("click_count_month").reset_index()
    user_click_count["click_count_month"] = user_click_count["click_count_month"] / MONTH
    t_user = t_user.merge(user_click_count, on=["uid"], how="left")
    # 每个人点击页面种数，整合到user表中
    user_click_class = t_click.groupby(["uid"])["pid"].nunique().rename("click_class_month").reset_index()
    t_user = t_user.merge(user_click_class, on=["uid"], how="left")
    t_user.fillna(0)
    return t_user


def user_feature(MONTH):
    #  用户ID，点击时间，点击页面，页面参数
    click = pd.read_csv("./data/JDloan/t_click.csv")
    #  用户ID，借款时间，借款金额，分期期数
    loan = pd.read_csv("./data/JDloan/t_loan.csv")
    #  用户ID，统计月份，借款总额
    loan_sum = pd.read_csv("./data/JDloan/t_loan_sum.csv")
    #  用户ID，购买时间，价格，数量，品类ID，优惠金额
    order = pd.read_csv("./data/JDloan/t_order.csv")
    #  用户ID，年龄段，性别，用户激活日期，初始额度
    user = pd.read_csv("./data/JDloan/t_user.csv")
    user = order_feature(order, user, MONTH)
    user = loan_feature(loan, user, MONTH)
    user = click_feature(click, user, MONTH)
    user["limit"] = user["limit"].apply(get_amount)
    user = user.merge(loan_sum, on=["uid"], how="left")
    # 平均每月购物-贷款
    user["diff_mean"] = user["price_sum_month"] - user["loan_sum_month"]
    # 总额度 - 初始金额
    user["diff"] = user["loan_sum_month"] * MONTH - user["limit"]
    user = user.fillna(0)
    user["loan_sum"] = user["loan_sum"].apply(get_amount)
    return user


def model(MONTH):
    user = user_feature(MONTH)
    x = user.drop(['uid', "loan_sum", "active_date", "month"], axis=1)
    y = user["loan_sum"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)
    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train.values.reshape(-1, 1))
    y_test = std_y.transform(y_test.values.reshape(-1, 1))

    rd = Ridge(alpha=1.0)  # alpha是超参数，可以取的值为0-1之间的小数，或者是1-10之间的整数
    rd.fit(x_train, y_train)  # 岭回归具有正则化，可以减少过拟合的现象
    print("岭回归系数", rd.coef_)
    y_rd_predict = std_y.inverse_transform(rd.predict(x_test))  # 结果逆标准化
    print("测试集预测的贷款额度", y_rd_predict)
    print("岭回归均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_rd_predict))
    joblib.dump(rd, "./rd_jdloan.pkl")
    return std_x, std_y


def main(MONTH):
    std_x, std_y = model(MONTH)
    user = user_feature(MONTH + 1)
    x = user.drop(['uid', "loan_sum", "active_date", "month"], axis=1)
    x = std_x.transform(x)
    clf = joblib.load("./rd_jdloan.pkl")
    y = std_y.inverse_transform(clf.predict(x))
    user["12_loan_sum"] = pd.DataFrame(y)
    user_save = user[["uid", "12_loan_sum"]]
    user_save.loc[user_save["12_loan_sum"] < 0, "12_loan_sum"] = 0
    user_save.to_csv("./user_12_loan.csv", index=False)


if __name__ == '__main__':
    main(3)  # 训练集是3个月的内容，测试集是4个月的内容

from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

# X = np.array([
#               [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [1, 2], [3, 4], [5, 6], [7, 8],
#               [9, 10], [11, 12], [13, 14], [15, 16]
#               ])
# y = np.array([0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1])
#
# # 分割训练集、测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# print("X_train=", X_train, " ,y_train=", y_train)
# print("X_test=", X_test, " ,y_test=", y_test)
#
#
# skf = StratifiedKFold(n_splits=4)
# for train_index, test_index in skf.split(X, y):
#     # print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     print("X_train=", X_train, " ,y_train=", y_train)
#     print("X_test=", X_test, " ,y_test=", y_test)

def data_clear(zhaban_data):
    # 对含有“price”字样的特征列，计算其与当日收盘价的涨跌幅
    price_feature = [i for i in zhaban_data.columns if 'price' in i]
    for feature in price_feature:
        zhaban_data[feature+'_ratio'] = (zhaban_data[feature] - zhaban_data['close']) / zhaban_data['close']

    # 对含有“vol”字样的特征列，计算其与当日成交量的涨跌幅
    vol_feature = [i for i in zhaban_data.columns if '_vol' in i]
    for feature in vol_feature:
        zhaban_data[feature+'_ratio'] = (zhaban_data[feature] - zhaban_data['vol']) / zhaban_data['vol']

    # 计算label标签列
    zhaban_data['is_zhaban'] = zhaban_data['zdt'].map(lambda x: 1 if x != '涨停' else 0)

    # 使用0替代缺失值
    zhaban_data = zhaban_data.fillna(0)
    return zhaban_data

def test_zhabandata():
    zhaban_data = pd.read_csv(
        "D:/pythonworkspace/sklearn_example_experiment/tensorflow_dnn_sklearn_gridsearch/zhaban_data/zhaban_feature.csv",
        index_col=0)

    # 数据预处理，如将缺失值转为0，将特征绝对值转化为比例值
    zhaban_data = data_clear(zhaban_data)

    # 只选取含有“ratio”字样的特征列作为最终的特征
    ratio_feature = [i for i in zhaban_data.columns if 'ratio' in i]

    X = np.array(zhaban_data[ratio_feature])
    X = np.nan_to_num(X)
    y = np.array(zhaban_data['is_zhaban'])
    skf = StratifiedKFold(n_splits=4)
    for train_index, test_index in skf.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print("X_train=", X_train, " ,y_train=", y_train)
        print("X_test=", X_test, " ,y_test=", y_test)

test_zhabandata()
from dnn_classifier import DNNClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

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

if __name__ == '__main__':
    zhaban_data = pd.read_csv("D:/pythonworkspace/sklearn_example_experiment/tensorflow_dnn_sklearn_gridsearch/zhaban_data/zhaban_feature.csv",
                          index_col=0)

    # 数据预处理，如将缺失值转为0，将特征绝对值转化为比例值
    zhaban_data = data_clear(zhaban_data)

    # 只选取含有“ratio”字样的特征列作为最终的特征
    ratio_feature = [i for i in zhaban_data.columns if 'ratio' in i]

    X = np.array(zhaban_data[ratio_feature])
    X = np.nan_to_num(X)
    y = np.array(zhaban_data['is_zhaban'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    # skf = StratifiedKFold(n_splits=4)
    #
    # # use randomized search to find the best hyperparameters
    # parameter_distributions = {
    #     'n_hidden_layers': [3, 4, 5],
    #     'n_neurons': [40, 50, 100],
    #     'batch_size' : [64, 128],
    #     'learning_rate': [0.01, 0.005, 0.001],
    #     'activation': [tf.nn.elu, tf.nn.relu],
    #     'max_checks_without_progress': [20, 30],
    #     'batch_norm_momentum': [None, 0.9],
    #     'dropout_rate': [None, 0.5]
    # }
    #
    # best_loss = np.float("inf")
    # best_paras = None
    # for hidden_layers in parameter_distributions['n_hidden_layers']:
    #     for neurons in parameter_distributions['n_neurons']:
    #         for batch_size in parameter_distributions['batch_size']:
    #             for learning_rate in parameter_distributions['learning_rate']:
    #                 for activation in parameter_distributions['activation']:
    #                     for batch_norm_momentum in parameter_distributions['batch_norm_momentum']:
    #                         for dropout_rate in parameter_distributions['dropout_rate']:
    #                             for train_index, test_index in skf.split(X_train, y_train):
    #                                 X_train_train, X_train_test = X[train_index], X[test_index]
    #                                 y_train_train, y_train_test = y[train_index], y[test_index]
    #                                 dnn = DNNClassifier(show_progress=1000, random_state=42, n_hidden_layers=hidden_layers
    #                                                     , n_neurons=neurons, batch_size=batch_size, learning_rate=learning_rate
    #                                                     , activation=activation, batch_norm_momentum=batch_norm_momentum
    #                                                     , dropout_rate=dropout_rate)
    #                                 val_loss = dnn.fit(X_train, y_train, 1000, X_train_test, y_train_test)
    #                                 if val_loss < best_loss:
    #                                     best_loss = val_loss
    #                                     # best_paras = dnn._get_model_parameters()
    #                                     dnn.save("models/grid_best_model")
    #                                 print("-----best_loss:" + str(best_loss))
    #                                 predictions = dnn.predict_metric_acc(X_test, y_test)
    #                                 print("predictions:" + str(predictions))
    #                                 print("n_hidden_layers: ", str(hidden_layers), " n_neurons: ", str(neurons), " batch_size: ", str(batch_size)
    #                                       , " learning_rate: ", str(learning_rate), " activation: " ,str(activation), " batch_norm_momentum: ", str(batch_norm_momentum)
    #                                       , " dropout_rate: ", str(dropout_rate))



    dnn = DNNClassifier(tensorboard_logdir="tensorboard_stats", n_hidden_layers=5, n_neurons=100, batch_size=64, learning_rate=0.01,
                        activation=tf.nn.elu, batch_norm_momentum=None, dropout_rate=None, random_state=42)
    dnn.fit(X_train, y_train, n_epochs=1000)
    predictions = dnn.predict_metric_acc(X_test, y_test)
    print("predictions:" + str(predictions))
    #
    # prediction = dnn.predict(X_test)
    # print("Score on test : {:.2f}%".format(accuracy_score(y_test, prediction) * 100))

    # dnn = DNNClassifier(show_progress=20, random_state=42)

    # use randomized search to find the best hyperparameters
    # parameter_distributions = {
    #     'n_hidden_layers': [3, 4, 5],
    #     'n_neurons': [40, 50, 100],
    #     'batch_size' : [64, 128],
    #     'learning_rate': [0.01, 0.005],
    #     'activation': [tf.nn.elu, tf.nn.relu],
    #     'max_checks_without_progress': [20, 30],
    #     'batch_norm_momentum': [None, 0.9],
    #     'dropout_rate': [None, 0.5]
    # }

    # parameter_distributions = {
    #     'n_hidden_layers': [5],
    #     'n_neurons': [100],
    #     'batch_size' : [128],
    #     'learning_rate': [0.005],
    #     'activation': [tf.nn.relu],
    #     'max_checks_without_progress': [30],
    #     'batch_norm_momentum': [0.9],
    #     'dropout_rate': [0.5]
    # }
    #
    # random_search = RandomizedSearchCV(dnn, parameter_distributions, n_iter=1, scoring='accuracy', verbose=2)
    # random_search.fit(X_train, y_train, n_epochs=1000)
    # best_params = random_search.best_params_
    # print("------best_params=" + best_params)
    # best_estimator = random_search.best_estimator_
    # mnist_predictions = best_estimator.predict(X_test)
    # print("Accuracy on the test set: {:.2f}".format(accuracy_score(y_test, mnist_predictions) * 100))
    # best_estimator.save("models/grid_best_model")

    # parameter_distributions = {
    #     'n_hidden_layers': [3, 4, 5],
    #     'n_neurons': [40, 50, 100],
    #     'batch_size' : [64, 128],
    #     'learning_rate': [0.01, 0.005, 0.001],
    #     'activation': [tf.nn.elu, tf.nn.relu],
    #     'max_checks_without_progress': [20, 30],
    #     'batch_norm_momentum': [None, 0.9],
    #     'dropout_rate': [None, 0.5]
    # }
    # 最好的参数组合: n_hidden_layers:  5  n_neurons:  100  batch_size:  64  learning_rate:  0.01  activation:  <function elu at 0x00000159D692F8C8>  batch_norm_momentum:  None  dropout_rate:  None
    # 误差: -----best_loss:0.260441
    # 正确率: -----predictions:0.760031471282
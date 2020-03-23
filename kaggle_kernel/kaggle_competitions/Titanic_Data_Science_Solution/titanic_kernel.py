import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')
test_PassengerId = test_df.PassengerId
# combine = [train_df, test_df]

# 删除感觉没用的列信息
train_df = train_df.drop(['Ticket', 'Cabin', 'PassengerId'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin', 'PassengerId'], axis=1)
combine = [train_df, test_df]

# 加入新的特征列
for dataset in combine:
    # 抽取Title列
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    # 将很少出现的title信息并成一类
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    # 对Title信息标准化处理
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    # 新建家庭大小列
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    # 是否独自一人
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1


# 特征encode
title_mapping = {'Mr': 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
sex_mapping = {'female': 1, 'male': 0}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Sex'] = dataset['Sex'].map(sex_mapping).astype(int)
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    # 对年龄列进行分桶操作 16/32/48/64，由'Age'列cut(5)得到
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4
    # 对Fare列进行分桶操作，7.91/14.454/31由'Fare'列qcut(4)得到
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare']

train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]


# 缺失值填充
guess_ages = np.zeros((2, 3))
freq_port = train_df['Embarked'].dropna().mode()[0]
for dataset in combine:
    # 用Sex、Pclass对应的median值填充age空值
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
            guess_ages[i, j] = guess_df.median()
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset['Age'].isnull()) & (dataset['Sex'] == i) & (dataset['Pclass'] == j+1), 'Age'] = guess_ages[i, j]
    dataset['Age'] = dataset['Age'].astype(int)
    dataset['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    dataset['Title'] = dataset['Title'].fillna(0)

# 构造新特征
# for dataset in combine:
#     dataset['Age*Class'] = dataset['Age'] * dataset['Pclass']

# 删除中间过程特征
train_df = train_df.drop(['Parch', 'SibSp'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp'], axis=1)

# from sklearn.model_selection import train_test_split
# X_train, X_val, y_train, y_val = train_test_split(train_df.drop('Survived', axis=1), train_df['Survived'], test_size=0.3, random_state=42)

X_train = train_df.drop('Survived', axis=1)
y_train = train_df['Survived']
X_test = test_df

# 对特征进行One_hot编码
from sklearn.preprocessing import OneHotEncoder
one_hot = OneHotEncoder(categories='auto', handle_unknown='ignore')
one_hot.fit(X_train)
X_train = one_hot.transform(X_train).toarray()
# X_val = one_hot.transform(X_val).toarray()
X_test = one_hot.transform(X_test).toarray()

# 特征选择
# from sklearn.feature_selection import SelectFromModel
# from sklearn.svm import LinearSVC
# clf = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train, y_train)
# model = SelectFromModel(clf, prefit=True)
# print("before feature select: " + str(X_train.shape))
# X_train = model.transform(X_train)
# print("after feature select: " + str(X_train.shape))
# X_test = model.transform(X_test)
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
#
# model = SelectKBest(chi2, k=18)
# model.fit(X_train, y_train)
# print("before feature select: " + str(X_train.shape))
# X_train = model.transform(X_train)
# print("after feature select: " + str(X_train.shape))
# X_test = model.transform(X_test)


# 建模
# 模型1：LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()
tuned_parameters = {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid = GridSearchCV(logistic, tuned_parameters, cv=5)
grid.fit(X_train, y_train)
best_est = grid.best_estimator_
logistic_Y_train_predic = best_est.predict(X_train)
logistic_Y_test_predic = best_est.predict(X_test)
acc_log_score = grid.best_score_
print("logistic best para :" + str(grid.best_params_))
print("logistic best score :" + str(grid.best_score_))
# logistic.fit(X_train, y_train)
# Y_predic = logistic.predict(X_test)
# acc_log = round(logistic.score(X_val, y_val), 2)
output = pd.DataFrame({'PassengerId': test_PassengerId,
                       'Survived': logistic_Y_test_predic})
output.to_csv('./data/logistic_submission.csv', index=False)
# 系数分析
# coeff_df = pd.DataFrame(X_train.columns, columns=['Feature'])
# coeff_df['Correlation'] = pd.Series(logistic.coef_[0])

# 模型2：
from sklearn.svm import SVC
svc = SVC()
tuned_parameters = {'kernel': ['rbf', 'linear'], 'gamma': [1e-2, 1e-3, 1e-4], 'C': [1, 10, 100, 1000]}
grid = GridSearchCV(svc, tuned_parameters, cv=5)
grid.fit(X_train, y_train)
best_est = grid.best_estimator_
svc_Y_train_predic = best_est.predict(X_train)
svc_Y_test_predic = best_est.predict(X_test)
acc_score = grid.best_score_
print("SVC best para :" + str(grid.best_params_))
print("SVC best score :" + str(grid.best_score_))
# svc.fit(X_train, y_train)
# Y_predic = svc.predict(X_test)
# acc_svc = round(svc.score(X_val, y_val), 2)
output = pd.DataFrame({'PassengerId': test_PassengerId,
                       'Survived': svc_Y_test_predic})
output.to_csv('./data/svm_submission.csv', index=False)

# 建模3
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
tuned_parameters = {'n_neighbors': [2, 3, 5], 'algorithm': ['auto', 'ball_tree', 'kd_tree'], 'leaf_size': [10, 30, 50]}
grid = GridSearchCV(knn, tuned_parameters, cv=5)
grid.fit(X_train, y_train)
best_est = grid.best_estimator_
knn_Y_train_predic = best_est.predict(X_train)
knn_Y_test_predic = best_est.predict(X_test)
knn_score = grid.best_score_
print("SVC best para :" + str(grid.best_params_))
print("SVC best score :" + str(grid.best_score_))
# knn.fit(X_train, y_train)
# Y_predic = knn.predict(X_test)
# acc_knn = round(knn.score(X_val, y_val), 2)
output = pd.DataFrame({'PassengerId': test_PassengerId,
                       'Survived': knn_Y_test_predic})
output.to_csv('./data/knn_submission.csv', index=False)

# 建模4
from sklearn.naive_bayes import GaussianNB
gauss_bayes = GaussianNB()
tuned_parameters = {'var_smoothing': [1e-9]}
grid = GridSearchCV(gauss_bayes, tuned_parameters, cv=5)
grid.fit(X_train, y_train)
best_est = grid.best_estimator_
gauss_bayes_Y_train_predic = best_est.predict(X_train)
gauss_bayes_Y_test_predic = best_est.predict(X_test)
gauss_bayes_score = grid.best_score_
print("SVC best para :" + str(grid.best_params_))
print("SVC best score :" + str(grid.best_score_))
# Y_predic = gauss_bayes.predict(X_test)
# acc_bayes = round(gauss_bayes.score(X_val, y_val), 2)
output = pd.DataFrame({'PassengerId': test_PassengerId,
                       'Survived': gauss_bayes_Y_test_predic})
output.to_csv('./data/bayes_submission.csv', index=False)

# 建模5
from sklearn.linear_model import Perceptron
perceptron = Perceptron()
tuned_parameters = {'alpha': [0.0001]}
grid = GridSearchCV(perceptron, tuned_parameters, cv=5)
grid.fit(X_train, y_train)
best_est = grid.best_estimator_
perceptron_Y_train_predic = best_est.predict(X_train)
perceptron_Y_test_predic = best_est.predict(X_test)
perceptron_score = grid.best_score_
print("SVC best para :" + str(grid.best_params_))
print("SVC best score :" + str(grid.best_score_))
# perceptron.fit(X_train, y_train)
# Y_predic = perceptron.predict(X_test)
# acc_perceptron = round(perceptron.score(X_val, y_val), 2)
output = pd.DataFrame({'PassengerId': test_PassengerId,
                       'Survived': perceptron_Y_test_predic})
output.to_csv('./data/linear_svc_submission.csv', index=False)

# 建模6 Linear SVC
# from sklearn.svm import LinearSVC
# linear_svc = LinearSVC()
# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                      'C': [1, 10, 100, 1000]},
#                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
# grid = GridSearchCV(linear_svc, tuned_parameters, cv=5)
# grid.fit(X_train, y_train)
# best_est = grid.best_estimator_
# Y_predic = best_est.predict(X_test)
# acc_linear_svc = grid.best_score_
# print("SVC best para :" + str(grid.best_params_))
# linear_svc.fit(X_train, y_train)
# Y_predic = linear_svc.predict(X_test)
# acc_linear_svc = round(linear_svc.score(X_val, y_val), 2)
# output = pd.DataFrame({'PassengerId': test_PassengerId,
#                        'Survived': Y_predic})
# output.to_csv('./data/linear_svc_submission.csv', index=False)

# 建模7 Stochastic Gradient Down
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier()
tuned_parameters = {'alpha': (0.001, 0.0001, 0.00001), 'penalty': ('l2', 'elasticnet')}
grid = GridSearchCV(sgd, tuned_parameters, cv=5)
grid.fit(X_train, y_train)
best_est = grid.best_estimator_
sgd_Y_train_predic = best_est.predict(X_train)
sgd_Y_test_predic = best_est.predict(X_test)
acc_sgd = grid.best_score_
print("sgd best para :" + str(grid.best_params_))
print("sgd best score :" + str(grid.best_score_))
# sgd.fit(X_train, y_train)
# Y_predic = sgd.predict(X_test)
# acc_sgd = round(sgd.score(X_val, y_val), 2)
output = pd.DataFrame({'PassengerId': test_PassengerId,
                       'Survived': sgd_Y_test_predic})
output.to_csv('./data/sgd_submission.csv', index=False)

# 建模8 decision Tree
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
tuned_parameters = {'min_samples_split': (2, 3, 5)}
grid = GridSearchCV(decision_tree, tuned_parameters, cv=5)
grid.fit(X_train, y_train)
best_est = grid.best_estimator_
decision_tree_Y_train_predic = best_est.predict(X_train)
decision_tree_Y_test_predic = best_est.predict(X_test)
decision_tree_score = grid.best_score_
print("sgd best para :" + str(grid.best_params_))
print("sgd best score :" + str(grid.best_score_))
# decision_tree.fit(X_train, y_train)
# Y_predic = decision_tree.predict(X_test)
# acc_decision_tree = round(decision_tree.score(X_val, y_val), 2)
output = pd.DataFrame({'PassengerId': test_PassengerId,
                       'Survived': decision_tree_Y_test_predic})
output.to_csv('./data/decision_tree_submission.csv', index=False)

# 建模9 Random Forest
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier()
# random_forest.fit(X_train, y_train)
# acc_random_forest = round(random_forest.score(X_val, y_val), 2)
# Y_predic = random_forest.predict(X_test)
tuned_parameters = {'max_depth': (2, 5, 10), 'n_estimators': (10, 50, 100)}
grid = GridSearchCV(random_forest, tuned_parameters, cv=5)
grid.fit(X_train, y_train)
best_est = grid.best_estimator_
random_forest_Y_train_predic = best_est.predict(X_train)
random_forest_Y_test_predic = best_est.predict(X_test)
acc_random_forest = grid.best_score_
print("random forest best para :" + str(grid.best_params_))
print("random forest best score :" + str(grid.best_score_))
# Save test predictions to file
output = pd.DataFrame({'PassengerId': test_PassengerId,
                       'Survived': random_forest_Y_test_predic})
output.to_csv('./data/random_forest_submission.csv', index=False)

X_train = np.concatenate((logistic_Y_train_predic.reshape(-1, 1), svc_Y_train_predic.reshape(-1, 1), knn_Y_train_predic.reshape(-1, 1),
                          gauss_bayes_Y_train_predic.reshape(-1, 1), perceptron_Y_train_predic.reshape(-1, 1), sgd_Y_train_predic.reshape(-1, 1),
                          decision_tree_Y_train_predic.reshape(-1, 1), random_forest_Y_train_predic.reshape(-1, 1)), axis=1)
X_test = np.concatenate((logistic_Y_test_predic.reshape(-1, 1), svc_Y_test_predic.reshape(-1, 1), knn_Y_test_predic.reshape(-1, 1),
                         gauss_bayes_Y_test_predic.reshape(-1, 1), perceptron_Y_test_predic.reshape(-1, 1), sgd_Y_test_predic.reshape(-1, 1),
                         decision_tree_Y_test_predic.reshape(-1, 1), random_forest_Y_test_predic.reshape(-1, 1)), axis=1)

# 建模9 xgboost
from xgboost import XGBClassifier
gbm = XGBClassifier(
    #learning_rate = 0.02,
    n_estimators= 2000,
    max_depth= 4,
    min_child_weight= 2,
    #gamma=1,
    gamma=0.9,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread= -1,
    scale_pos_weight=1).fit(X_train, y_train)
predictions = gbm.predict(X_test)
print("gbm score: " + str(gbm.score(X_train, y_train)))

# Generate Submission File
StackingSubmission = pd.DataFrame({'PassengerId': test_PassengerId, 'Survived': predictions})
StackingSubmission.to_csv("./data/StackingSubmission.csv", index=False)

# tmp_dct = {'Model': ['SVM', 'KNN', 'LogisticRegression', 'RandomForest', 'Naive Bayes', 'Perceptron', 'SGDClassifier', 'Linear SVC', 'Decision Tree'],
#            'accuracy': [acc_svc, acc_knn, acc_log, acc_random_forest, acc_bayes, acc_perceptron, acc_sgd, acc_linear_svc, acc_decision_tree]}
# result = pd.DataFrame(tmp_dct)
# result.sort_values(by='accuracy', ascending=False, inplace=True)

# kaggle比赛提交文件的格式
# model = RandomForestRegressor(n_estimators=100, random_state=0)
# model.fit(OH_X_train, y_train)
# preds_test = model.predict(OH_X_test)
#
# # Save test predictions to file
# output = pd.DataFrame({'Id': OH_X_test.index,
#                        'SalePrice': preds_test})
# output.to_csv('submission.csv', index=False)

# train_df.info()
# train_df.describe()
# 类别型变量描述统计信息
# train_df.describe(include=['o'])

# 可以验证假设
# "不同Pclass的存活率不一样，贵族(Pclass=1)有较高的存活率"
# train_df[['Pclass', 'Survived']].groupby('Pclass', as_index=False).mean().sort_values(by='Survived', ascending=False)

# 可以验证假设
# "女性存活率较高"
# train_df[['Sex', 'Survived']].groupby('Sex', as_index=False).mean().sort_values(by='Survived', ascending=False)

# 可以验证假设
# "SibSp"、"Parch"与存活率相关性不大。可以考虑从"SibSp"、"Parch"衍生出一些新变量，如家庭人员数：SibSp + Parch + 1
# train_df[['SibSp', 'Survived']].groupby('SibSp', as_index=False).mean().sort_values(by='Survived', ascending=False)
#    SibSp  Survived
# 1      1  0.535885
# 2      2  0.464286
# 0      0  0.345395
# 3      3  0.250000
# 4      4  0.166667
# 5      5  0.000000
# 6      8  0.000000
# train_df[['Parch', 'Survived']].groupby('Parch', as_index=False).mean().sort_values(by='Survived', ascending=False)
#    Parch  Survived
# 3      3  0.600000
# 1      1  0.550847
# 2      2  0.500000
# 0      0  0.343658
# 5      5  0.200000
# 4      4  0.000000
# 6      6  0.000000

# "年龄与存活率"直方图
# g = sns.FacetGrid(train_df, col='Survived')
# g.map(plt.hist, 'Age', bins=20)

# "Pclass、年龄与存活率"直方图
# grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=0.5, bins=20)
# grid.add_legend()


# 探索"Embarked、Pclass、Survived、Sex"关系
# grid = sns.FacetGrid(train_df, row='Embarked')
# grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex')
# grid.add_legend()

# 探索"Embarked、Survived、Sex、Fare"关系
# grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
# grid.map(sns.barplot, 'Sex', 'Fare')
# grid.add_legend()

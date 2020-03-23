import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, GridSearchCV, learning_curve
import matplotlib.pyplot as plt
import string


def extract_surname(data):
    families = []
    for i in range(len(data)):
        name = data.iloc[i]
        if '(' in name:
            name_no_bracket = name.split('(')[0]
        else:
            name_no_bracket = name
        family = name_no_bracket.split(',')[0]
        title = name_no_bracket.split(',')[1].strip().split(' ')[0]
        for c in string.punctuation:
            family = family.replace(c, '').strip()
        families.append(family)
    return families


# 创建基于target计算得到的feature
def create_target_based_feature(dataset, train_len):
    dataset['Family'] = extract_surname(dataset['Name'])
    dataset['Ticket_Frequency'] = dataset.groupby('Ticket')['Ticket'].transform('count')
    df_train = dataset[:train_len]
    df_test = dataset[train_len:]
    mean_survival_rate = np.mean(df_train['Survived'])

    train_family_survival_rate = []
    train_family_survival_rate_NA = []
    test_family_survival_rate = []
    test_family_survival_rate_NA = []

    non_unique_families = [x for x in df_train['Family'].unique() if x in df_test['Family'].unique()]
    non_unique_tickets = [x for x in df_train['Ticket'].unique() if x in df_test['Ticket'].unique()]

    df_family_survival_rate = df_train.groupby('Family')['Survived', 'Family', 'Fsize'].median()
    df_ticket_survival_rate = df_train.groupby('Ticket')['Survived', 'Ticket', 'Ticket_Frequency'].median()

    family_rates = {}
    ticket_rates = {}

    for i in range(len(df_family_survival_rate)):
        # Checking a family exists in both training and test set, and has members more than 1
        if df_family_survival_rate.index[i] in non_unique_families and df_family_survival_rate.iloc[i, 1] > 1:
            family_rates[df_family_survival_rate.index[i]] = df_family_survival_rate.iloc[i, 0]

    for i in range(len(df_ticket_survival_rate)):
        # Checking a ticket exists in both training and test set, and has members more than 1
        if df_ticket_survival_rate.index[i] in non_unique_tickets and df_ticket_survival_rate.iloc[i, 1] > 1:
            ticket_rates[df_ticket_survival_rate.index[i]] = df_ticket_survival_rate.iloc[i, 0]

    for i in range(len(df_train)):
        if df_train['Family'][i] in family_rates:
            train_family_survival_rate.append(family_rates[df_train['Family'][i]])
            train_family_survival_rate_NA.append(1)
        else:
            train_family_survival_rate.append(mean_survival_rate)
            train_family_survival_rate_NA.append(0)

    for i in range(len(df_test)):
        if df_test['Family'].iloc[i] in family_rates:
            test_family_survival_rate.append(family_rates[df_test['Family'].iloc[i]])
            test_family_survival_rate_NA.append(1)
        else:
            test_family_survival_rate.append(mean_survival_rate)
            test_family_survival_rate_NA.append(0)

    df_train['Family_Survival_Rate'] = train_family_survival_rate
    df_train['Family_Survival_Rate_NA'] = train_family_survival_rate_NA
    df_test['Family_Survival_Rate'] = test_family_survival_rate
    df_test['Family_Survival_Rate_NA'] = test_family_survival_rate_NA

    train_ticket_survival_rate = []
    train_ticket_survival_rate_NA = []
    test_ticket_survival_rate = []
    test_ticket_survival_rate_NA = []

    for i in range(len(df_train)):
        if df_train['Ticket'][i] in ticket_rates:
            train_ticket_survival_rate.append(ticket_rates[df_train['Ticket'][i]])
            train_ticket_survival_rate_NA.append(1)
        else:
            train_ticket_survival_rate.append(mean_survival_rate)
            train_ticket_survival_rate_NA.append(0)

    for i in range(len(df_test)):
        if df_test['Ticket'].iloc[i] in ticket_rates:
            test_ticket_survival_rate.append(ticket_rates[df_test['Ticket'].iloc[i]])
            test_ticket_survival_rate_NA.append(1)
        else:
            test_ticket_survival_rate.append(mean_survival_rate)
            test_ticket_survival_rate_NA.append(0)

    df_train['Ticket_Survival_Rate'] = train_ticket_survival_rate
    df_train['Ticket_Survival_Rate_NA'] = train_ticket_survival_rate_NA
    df_test['Ticket_Survival_Rate'] = test_ticket_survival_rate
    df_test['Ticket_Survival_Rate_NA'] = test_ticket_survival_rate_NA
    dataset.loc[:train_len, 'Survival_Rate'] = (df_train['Ticket_Survival_Rate'] + df_train['Family_Survival_Rate']) / 2
    dataset.loc[train_len:, 'Survival_Rate'] = (df_test['Ticket_Survival_Rate'] + df_test['Family_Survival_Rate']) / 2
    dataset.loc[:train_len, 'Survival_Rate_NA'] = (df_train['Ticket_Survival_Rate_NA'] + df_train['Family_Survival_Rate_NA']) / 2
    dataset.loc[train_len:, 'Survival_Rate_NA'] = (df_test['Ticket_Survival_Rate_NA'] + df_test['Family_Survival_Rate_NA']) / 2
    dataset.drop(labels=['Family'], axis=1, inplace=True)
    return dataset['Survival_Rate'], dataset['Survival_Rate_NA']


# 2 Load and check data
# 2.1 Load data
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')
test_PassengerId = test_df.PassengerId
# combine = [train_df, test_df]
# 2.2 Outlier detection
def detect_outliers(df, n , features):
    """
    Takes a dataframe df of features and returns a list of the indices corresponding to the observations
    containing more than n outliers according to the Tukey mathod.
    """
    outlier_indices = []
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col], 75)
        # Interquartile range(IQR)
        IQR = Q3 - Q1
        # outlier step
        outlier_step = 1.5 * IQR
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)
    return multiple_outliers
# detect outliers from Age, SibSp, Parch and Fare
Outliers_to_drop = detect_outliers(train_df, 2, ['Age', 'SibSp', 'Parch', 'Fare'])
train_df = train_df.drop(Outliers_to_drop, axis=0).reset_index(drop=True)
# 2.3 joining train and test set (join train and test datasets in order to obtain the same number of features during
#     categorical conversion)
train_len = len(train_df)
dataset = pd.concat(objs=[train_df, test_df], axis=0).reset_index(drop=True)
# 2.4 check for null and missing values
# fill empty and NaNs values with NaN
dataset = dataset.fillna(np.nan)
# Check for Null values
print(dataset.isnull().sum())

# 3 Feature analysis
# 3.1 Numerical values
# g = sns.heatmap(train_df[['Survived', 'SibSp', 'Parch', 'Age', 'Fare']].corr(), annot=True, fmt='.2f', cmap='coolwarm')
# feature: SibSp
# g = sns.factorplot(x='SibSp', y='Survived', data=train_df, kind='bar', size=6, palette='muted')
# g.despine(left=True)
# g = g.set_ylabels('survival probability')
# feature: Parch
# g = sns.factorplot(x='Parch', y='Survived', data=train_df, kind='bar', size=6, palette='muted')
# g.despine(left=True)
# g = g.set_ylabels('Survival probability')
# feature: Age
# g = sns.FacetGrid(train_df, col='Survived')
# g = g.map(sns.distplot, "Age")

# g = sns.kdeplot(train_df['Age'][(train_df['Survived'] == 0) & (train_df['Age'].notnull())], color='Red', shade=True)
# g = sns.kdeplot(train_df['Age'][(train_df['Survived'] == 1) & (train_df['Age'].notnull())], ax=g, color='Blue', shade=True)
# g.set_xlabel("Age")
# g.set_ylabel("Frequency")
# g = g.legend(["Not Survived", "Survived"])
# feature: Fare
print(dataset["Fare"].isnull().sum())
dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median()).astype(float)
# g = sns.distplot(dataset['Fare'], color='m', label="Skewness: %.2f" % (dataset['Fare'].skew(axis=0)))
# g = g.legend(loc='best')
dataset['Fare'] = dataset['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
# 3.1 Category values
# feature: Sex
# g = sns.barplot(x='Sex', y='Survived', data=train_df)
# g = g.set_ylabel("Survived Probability")
print(train_df[['Sex', 'Survived']].groupby('Sex').mean())
# feature: Pclass
# g = sns.factorplot(x='Pclass', y='Survived', data=train_df, kind='bar', size=6, palette='muted')
# g.despine(left=True)
# g = g.set_ylabels('survival probability')

# g = sns.factorplot(x='Pclass', y='Survived', hue='Sex', data=train_df, size=6, kind='bar', palette='muted')
# g.despine(left=True)
# g = g.set_ylabels('survival probability')
# feature: Embarked
print(dataset['Embarked'].isnull().sum())
most_occur = train_df['Embarked'].dropna().mode().values[0]
dataset['Embarked'] = dataset['Embarked'].fillna(most_occur)
# g = sns.factorplot(x='Embarked', y='Survived', data=train_df, size=6, kind='bar', palette='muted')
# g.despine(left=True)
# g = g.set_ylabels('survival probability')
# g = sns.factorplot("Pclass", col="Embarked", data=train_df, size=6, kind='count', palette='muted')
# g.despine(left=True)
# g = g.set_ylabels('Count')
# 4.Filling missing values
# 4.1 Age
# g = sns.factorplot(x='Sex', y='Age', data=dataset, kind='box')
# g = sns.factorplot(x='Sex', y='Age', hue='Pclass', data=dataset, kind='box')
# g = sns.factorplot(x='Parch', y='Age', data=dataset, kind='box')
# g = sns.factorplot(x='SibSp', y='Age', data=dataset, kind='box')
dataset['Sex'] = dataset['Sex'].map({'male': 0, 'female': 1})
# g = sns.heatmap(dataset[['Age', 'Sex', 'SibSp', 'Parch', 'Pclass']].corr(), cmap='BrBG', annot=True)
index_NaN_age = list(dataset['Age'][dataset['Age'].isnull()].index)
for i in index_NaN_age:
    age_med = dataset['Age'].median()
    age_pred = dataset['Age'][(dataset['SibSp'] == dataset.iloc[i]['SibSp']) &
                              (dataset['Parch'] == dataset.iloc[i]['Parch']) &
                              (dataset['Pclass'] == dataset.iloc[i]['Pclass'])].median()
    if not np.isnan(age_pred):
        dataset['Age'].iloc[i] = age_pred
    else:
        dataset['Age'].iloc[i] = age_med
# g = sns.factorplot(x='Survived', y='Age', data=train_df, kind='box')
# g = sns.factorplot(x='Survived', y='Age', data=train_df, kind='violin')
# 5. Feature engineering
# 5.0 Family size
dataset['Fsize'] = dataset['SibSp'] + dataset['Parch'] + 1
# g = sns.factorplot(x='Fsize', y='Survived', data=dataset)
# g = g.set_ylabels("survival probability")

dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)
dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if s == 2 else 0)
dataset['MedF'] = dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
dataset['LargeF'] = dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)
# g = sns.factorplot(x='Single', y='Survived', data=dataset, kind='bar')
# g = g.set_ylabels("survival probability")
# g = sns.factorplot(x='SmallF', y='Survived', data=dataset,kind='bar')
# g = g.set_ylabels("survival probability")
# g = sns.factorplot(x='MedF', y='Survived', data=dataset, kind='bar')
# g = g.set_ylabels("survival probability")
# g = sns.factorplot(x='LargeF', y='Survived', data=dataset, kind='bar')
# g = g.set_ylabels("survival probability")
# 5.1 target based features
dataset['Survival_Rate'], dataset['Survival_Rate_NA'] = create_target_based_feature(dataset, train_len)
# 5.2 Name/Title
dataset_title = [i.split(',')[1].split('.')[0].strip() for i in dataset['Name']]
dataset['Title'] = pd.Series(dataset_title)
print(dataset['Title'].head())
# g = sns.factorplot("Title", data=dataset, size=6, kind='count', palette='muted')
dataset['Title'] = dataset['Title'].replace(['Lady', 'the Countess', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset['Title'] = dataset['Title'].map({"Master": 0, "Miss": 1, "Ms": 1, "Mme": 1, "Mlle": 1, "Mrs": 1, "Mr": 2, "Rare": 3})
dataset['Title'] = dataset['Title'].astype(int)
# g = sns.factorplot("Title", data=dataset, size=6, kind='count', palette='muted')
# g = g.set_xticklabels(["Master", "Miss/Ms/Mme/Mlle/Mrs", "Mr", "Rare"])
# g = sns.factorplot(x="Title", y='Survived', data=dataset, size=6, kind='bar', palette='muted')
# g = g.set_xticklabels(["Master", "Miss/Ms/Mme/Mlle/Mrs", "Mr", "Rare"])
# g = g.set_ylabels("survival probability")
dataset.drop(labels=['Name'], axis=1, inplace=True)

dataset = pd.get_dummies(dataset, columns=['Title'])
dataset = pd.get_dummies(dataset, columns=['Embarked'], prefix='Em')
# 5.3 Cabin
print(dataset['Cabin'].head())
print(dataset['Cabin'].describe())
print(dataset['Cabin'].isnull().sum())
dataset['Cabin'] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin']])
# g = sns.factorplot(x='Cabin', y='Survived', data=dataset, kind='bar', order=['A','B','C','D','E','F','G','T','X'])
dataset = pd.get_dummies(dataset, columns=['Cabin'], prefix='Cabin')
# 5.4 ticket
print(dataset['Ticket'].head())
Ticket = []
for i in list(dataset['Ticket']):
    if not i.isdigit():
        Ticket.append(i.replace('.', '').replace('/', '').strip().split(' ')[0])
    else:
        Ticket.append('X')
dataset['Ticket'] = Ticket
print(dataset['Ticket'].head())
dataset = pd.get_dummies(dataset, columns=['Ticket'], prefix='T')

dataset['Pclass'] = dataset['Pclass'].astype('category')
dataset = pd.get_dummies(dataset, columns=['Pclass'], prefix='Pc')
dataset.drop(labels=['PassengerId'], axis=1, inplace=True)

# 6 MODELING
# 6.1 sample model
train = dataset[: train_len - 100]
val = dataset[train_len - 100: train_len]
test = dataset[train_len:]
test.drop(labels=['Survived'], axis=1, inplace=True)

train['Survived'] = train['Survived'].astype(int)
Y_train = train['Survived']
X_train = train.drop(labels=['Survived'], axis=1)
Y_val = val['Survived']
X_val = val.drop(labels=['Survived'], axis=1)

kfold = StratifiedKFold(n_splits=5)
# random_state = 2
# classifiers = []
# models_names = ['svc', 'DicisionTree', 'AdaBoost', 'RandomForest', 'ExtraTree', 'GradientBoosting', 'MultipleLayerPerceptron',
#                 'KNeighbors', 'LogisticRegression', 'LinearDiscriminantAnalysis']
# classifiers.append(SVC(random_state=random_state))
# classifiers.append(DecisionTreeClassifier(random_state=random_state))
# classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state), random_state=random_state, learning_rate=0.1))
# classifiers.append(RandomForestClassifier(random_state=random_state))
# classifiers.append(ExtraTreesClassifier(random_state=random_state))
# classifiers.append(GradientBoostingClassifier(random_state=random_state))
# classifiers.append(MLPClassifier(random_state=random_state))
# classifiers.append(KNeighborsClassifier())
# classifiers.append(LogisticRegression(random_state=random_state))
# classifiers.append(LinearDiscriminantAnalysis())

# cv_results = []
# for classifier in classifiers:
#     cv_results.append(cross_val_score(classifier, X=X_train, y=Y_train, scoring="accuracy", cv=kfold, n_jobs=4))
# cv_means = []
# cv_std = []
# for cv_result in cv_results:
#     cv_means.append(cv_result.mean())
#     cv_std.append(cv_result.std())
# cv_res = pd.DataFrame({"cv_mean": cv_means, "cv_std": cv_std, "algorithm": models_names})
# g = sns.barplot("cv_mean", "algorithm", data=cv_res, palette="Set3", orient="h", **{'xerr': cv_std})
# g.set_xlabel("mean accuracy")
# g = g.set_title("cross validation scores")
# 6.1.2 turn the model
# Adaboost
# DTC = DecisionTreeClassifier()
# adaDTC = AdaBoostClassifier(DTC, random_state=random_state)
# ada_params = {"base_estimator__criterion": ['gini', 'entropy'],
#               "base_estimator__splitter": ["best", "random"],
#               "algorithm": ['SAMME', 'SAMME.R'],
#               "n_estimators": [1, 2],
#               "learning_rate": [0.0001, 0.0005, 0.001, 0.005, 0.1, 0.5]}
# grid_ada_dtc = GridSearchCV(adaDTC, param_grid=ada_params, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)
# grid_ada_dtc.fit(X_train, Y_train)
# ada_best = grid_ada_dtc.best_estimator_
# print(grid_ada_dtc.best_score_)
# ExtraTree
# extc = ExtraTreesClassifier()
# ex_params = {"max_depth": [None],
#              "max_features": [1, 3, 10],
#              "min_samples_split": [2, 3, 10],
#              "min_samples_leaf": [1, 3, 10],
#              "bootstrap": [False],
#              "n_estimators": [100, 300],
#              "criterion": ['gini']}
# grid_ext_tree = GridSearchCV(extc, ex_params, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)
# grid_ext_tree.fit(X_train, Y_train)
# ext_tree_best = grid_ext_tree.best_estimator_
# print(grid_ext_tree.best_score_)
# RandomForestClassifier
random_forest = RandomForestClassifier()
rand_forest_paras = {"max_depth": [None],
                     "max_features": [1, 3, 10],
                     "min_samples_split": [2, 3, 10],
                     "min_samples_leaf": [1, 3, 10],
                     "bootstrap": [False],
                     "n_estimators": [100, 300],
                     "criterion": ['gini']}
grid_rand_forest = GridSearchCV(random_forest, rand_forest_paras, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)
grid_rand_forest.fit(X_train, Y_train)
rand_forest_best = grid_rand_forest.best_estimator_
print(grid_rand_forest.best_score_)
# 预测验证集中的数据并观察分类错误的样本有什么特性，提高模型的accuracy
val_predict = rand_forest_best.predict(X_val)
val_result = X_val.copy(deep=True)
val_result['survived_act'] = Y_val
val_result['survived_predict'] = val_predict
val_result.to_csv('./data/val.csv')
import eli5
from eli5.sklearn import PermutationImportance
perm = PermutationImportance(rand_forest_best, random_state=1).fit(X_val, Y_val)
eli5.show_weights(perm,  feature_names=X_val.columns.tolist())
print(eli5.format_as_text(eli5.explain_weights(perm,  feature_names=X_val.columns.tolist())))
importances = list(rand_forest_best.feature_importances_)
importances_dict = dict(zip(X_val.columns.tolist(), importances))
sort_imp = sorted(importances_dict.items(), key=lambda d: d[1], reverse=True)
for item in sort_imp:
    print('%s, %s' % (item[0], item[1]))
# # 观察错误分类样本的各特征贡献
# data_for_prediction = X_val.loc[791].values.reshape(1, -1)
# import shap
# k_explainer = shap.KernelExplainer(rand_forest_best.predict_proba, X_val)
# k_shap_values = k_explainer.shap_values(data_for_prediction)
# shap.initjs()
# shap.force_plot(k_explainer.expected_value[1], k_shap_values[1], data_for_prediction, feature_names=X_val.columns.tolist(), show=False, matplotlib=True)
# plt.savefig("./data/shap.png", dpi=150, bbox_inches='tight')
predictions = rand_forest_best.predict(test)
StackingSubmission = pd.DataFrame({'PassengerId': test_PassengerId, 'Survived': predictions})
StackingSubmission.to_csv("./data/randomforest_with_targetbased_feature_Submission.csv", index=False)
# Gradient boosting tunning
# gradient_boost = GradientBoostingClassifier()
# gradient_boost_paras = {'loss': ['deviance'],
#                         'n_estimators': [100, 200, 300],
#                         'learning_rate': [0.1, 0.05, 0.01],
#                         'max_depth': [4, 8],
#                         'min_samples_leaf': [100, 150],
#                         'max_features': [0.3, 0.1]}
# gradient_boost_grid = GridSearchCV(gradient_boost, gradient_boost_paras, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)
# gradient_boost_grid.fit(X_train, Y_train)
# gradient_boost_best = gradient_boost_grid.best_estimator_
# print(gradient_boost_grid.best_score_)
# SVC
# svc = SVC()
# svc_paras = {'kernel': ['rbf'],
#              'gamma': [0.001, 0.01, 0.1, 1],
#              'C': [1, 10, 50, 100, 500, 1000]}
# svc_grid = GridSearchCV(svc, svc_paras, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)
# svc_grid.fit(X_train, Y_train)
# svc_best = svc_grid.best_estimator_
# print(svc_grid.best_score_)
# 预测验证集中的数据并观察分类错误的样本有什么特性，提高模型的accuracy
# val_predict = svc_best.predict(X_val)
# val_result = X_val.copy(deep=True)
# val_result['survived_act'] = Y_val
# val_result['survived_predict'] = val_predict
# val_result.to_csv('./data/val.csv')
# # 特征重要性分析
# import eli5
# from eli5.sklearn import PermutationImportance
# perm = PermutationImportance(svc_best, random_state=1).fit(X_val, Y_val)
# print(eli5.format_as_text(eli5.explain_weights(perm,  feature_names=X_val.columns.tolist())))

# 6.1.3 plot learning curves
# def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=-1, train_size=np.linspace(0.1, 1.0, 5)):
#     """Generate a simple plot of the test and training learning curve"""
#     plt.figure()
#     plt.title(title)
#     if ylim is not None:
#         plt.ylim(*ylim)
#     plt.xlabel('Training examples')
#     plt.ylabel('Score')
#     train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_size)
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
#     plt.grid()
#
#     plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
#     plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='g')
#
#     plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
#     plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Test score')
#
#     plt.legend(loc='best')
#     return plt

# g = plot_learning_curve(rand_forest_best, "Random Forest learning curves", X_train, Y_train, cv=kfold)
# g = plot_learning_curve(ext_tree_best, "ExtraTrees learning curves", X_train, Y_train, cv=kfold)
# g = plot_learning_curve(svc_best, "SVC learning curves", X_train, Y_train, cv=kfold)
# g = plot_learning_curve(ada_best, "Adaboost learning curves", X_train, Y_train, cv=kfold)
# g = plot_learning_curve(gradient_boost_best, "GradientBoosting learning curves", X_train, Y_train, cv=kfold)

# 6.1.4 feature importance
# nrows = ncols = 2
# fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex="all", figsize=(15, 15))
# names_classifiers = [("AdaBoosting", ada_best), ("ExtraTrees", ext_tree_best), ('RandomForest', rand_forest_best),
#                      ('GradientBoosting', gradient_boost_best)]
# nclassifier = 0
# for row in range(nrows):
#     for col in range(ncols):
#         name = names_classifiers[nclassifier][0]
#         classifier = names_classifiers[nclassifier][1]
#         indices = np.argsort(classifier.feature_importances_)[::-1][:10]
#         g = sns.barplot(y=X_train.columns[indices][:10], x=classifier.feature_importances_[indices][:10], orient='h', ax=axes[row][col])
#         g.set_xlabel("Relative importance", fontsize=12)
#         g.set_ylabel("Features", fontsize=12)
#         g.tick_params(labelsize=9)
#         g.set_title(name + " feature importance")
#         nclassifier += 1

# test_survived_rand_forest = pd.Series(rand_forest_best.predict(test), name='Random Forest')
# test_survived_extra_trees = pd.Series(ext_tree_best.predict(test), name='Extra Trees')
# test_survived_svc = pd.Series(svc_best.predict(test), name="SVN")
# test_survived_adaboost = pd.Series(ada_best.predict(test), name="Adaboost")
# test_survived_gradient_boost = pd.Series(gradient_boost_best.predict(test), name="Gradient Boost")

# ensemble_results = pd.concat([test_survived_rand_forest, test_survived_extra_trees, test_survived_svc,
#                               test_survived_adaboost, test_survived_gradient_boost], axis=1)
# g = sns.heatmap(ensemble_results.corr(), annot=True)
# 6.2 Ensemble modeling
# 6.2.1 Combine models
# votingC = VotingClassifier(estimators=names_classifiers, voting='soft', n_jobs=4)
# votingC.fit(X_train, Y_train)
# 6.3 Prediction
# 6.3.1 Predict and Submit results
# predictions = votingC.predict(test)
# predictions = svc_best.predict(test)
# StackingSubmission = pd.DataFrame({'PassengerId': test_PassengerId, 'Survived': predictions})
# StackingSubmission.to_csv("./data/svn_v2_Submission.csv", index=False)


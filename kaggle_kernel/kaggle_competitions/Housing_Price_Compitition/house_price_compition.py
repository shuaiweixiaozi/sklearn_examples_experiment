import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

train_df = pd.read_csv('./data/train.csv')
# 因为目标变量有正偏度，需要进行对数转换(log)
train_df['SalePrice'] = train_df['SalePrice'].map(np.log1p)
test_df = pd.read_csv('./data/test.csv')
test_Id = test_df.Id
dataset = pd.concat(objs=[train_df, test_df], axis=0).reset_index(drop=True)
train_len = len(train_df)
# del dataset['SalePrice']
total_num = len(dataset)
miss_val_cols = []

# 查看数据整体情况
# dataset.info()
# dataset.describe()


# 查看以后发现有很多列空值占比很高、同一值的占比很高，删除这些列
def delete_null_or_mode_rat_high_cols(dataset):
    for col in dataset.columns:
        # 计算列空值占比
        null_num = dataset[col].isnull().sum()
        null_rat = null_num / total_num
        # todo 可以再细看一下，有值的数据中是否集中在某一两个值，这样的列可以将空值作为一种特殊值对待
        if null_rat >= 0.7:
            print("drop high null rat [%s]: %s" % (col, null_rat))
            del dataset[col]
            continue
        # 计算列中值是否集中在某个值，
        desc = dataset[col].astype(object).describe()
        mode_rat = desc['freq'] / total_num
        if mode_rat > 0.9:
            print("drop high mode rat [%s]: %s" % (col, mode_rat))
            del dataset[col]
            continue
        if null_rat > 0:
            miss_val_cols.append(col)
    return dataset

dataset = delete_null_or_mode_rat_high_cols(dataset)

# 查看各列的相关系数
# corr = dataset.corr()
# sns.heatmap(corr, cmap='BrBG', annot=True)

# 异常值处理
## https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python 从文中得到GrLivArea与SalePrice相关性很大，观察GrLivArea列的异常值
# dataset.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0, 800000))
# print(dataset.sort_values(by='GrLivArea', ascending=False)[:2])
# dataset = dataset.drop(train_df[train_df['Id'] == 1299].index)
# dataset = dataset.drop(train_df[train_df['Id'] == 524].index)
# train_len = train_len - 2

# 缺失值补全
## 标注的列，NA值有特殊的含义，需要特殊处理
## BsmtCond
dataset['BsmtCond'] = dataset['BsmtCond'].fillna('0')
dataset['BsmtExposure'] = dataset['BsmtExposure'].fillna('0')
dataset['BsmtFinType1'] = dataset['BsmtFinType1'].fillna('0')
dataset['BsmtFinType2'] = dataset['BsmtFinType2'].fillna('0')
dataset['BsmtQual'] = dataset['BsmtQual'].fillna('0')
dataset['FireplaceQu'] = dataset['FireplaceQu'].fillna('0')
dataset['GarageFinish'] = dataset['GarageFinish'].fillna('0')
dataset['GarageQual'] = dataset['GarageQual'].fillna('0')
dataset['GarageType'] = dataset['GarageType'].fillna('0')
dataset['MasVnrType'] = dataset['MasVnrType'].fillna('0')

for col in miss_val_cols:
    # 数值型变量，用均值填充
    if str(dataset[col].dtype) in ['int64', 'float64']:
        mean = dataset[col].median()
        dataset[col] = dataset[col].fillna(mean)
    # 类别型变量，用众数填充
    else:
        most_occur = dataset[col].dropna().mode().values[0]
        dataset[col] = dataset[col].fillna(most_occur)

# 特征编码
## category特征如果有明显的次序关系，用LabelEncode编码
dataset['BsmtCond'] = dataset['BsmtCond'].map({'0': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}).astype(int)
dataset['BsmtExposure'] = dataset['BsmtExposure'].map({'0': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}).astype(int)
dataset['BsmtFinType1'] = dataset['BsmtFinType1'].map({'0': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}).astype(int)
dataset['BsmtFinType2'] = dataset['BsmtFinType2'].map({'0': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}).astype(int)
dataset['BsmtQual'] = dataset['BsmtQual'].map({'0': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}).astype(int)
dataset['ExterCond'] = dataset['ExterCond'].map({'0': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}).astype(int)
dataset['ExterQual'] = dataset['ExterQual'].map({'0': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}).astype(int)
dataset['FireplaceQu'] = dataset['FireplaceQu'].map({'0': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}).astype(int)
dataset['GarageFinish'] = dataset['GarageFinish'].map({'0': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}).astype(int)
dataset['GarageQual'] = dataset['GarageQual'].map({'0': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}).astype(int)
dataset['HeatingQC'] = dataset['HeatingQC'].map({'0': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}).astype(int)
dataset['KitchenQual'] = dataset['KitchenQual'].map({'0': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}).astype(int)
dataset['LandContour'] = dataset['LandContour'].map({'0': 0, 'Low': 1, 'HLS': 2, 'Bnk': 3, 'Lvl': 4}).astype(int)
## category特征没有明显的次序关系，用OneHot编码
dataset = pd.get_dummies(dataset, columns=['BldgType', 'Condition1', 'Exterior1st', 'Exterior2nd',
            'Foundation', 'GarageType', 'HouseStyle', 'LotConfig', 'LotShape', 'MSZoning', 'MasVnrType', 'Neighborhood',
            'RoofStyle', 'SaleCondition', 'SaleType'])
dataset = delete_null_or_mode_rat_high_cols(dataset)

# 划分测试集训练集
train = dataset[: train_len - 1]
val = dataset[train_len - 1: train_len]
test = dataset[train_len:]
test.drop(labels=['SalePrice'], axis=1, inplace=True)

train['SalePrice'] = train['SalePrice'].astype(float)
Y_train = train['SalePrice']
from sklearn.utils.multiclass import type_of_target
print('------%s' % str(type_of_target(Y_train)))
X_train = train.drop(labels=['SalePrice'], axis=1)
Y_val = val['SalePrice']
X_val = val.drop(labels=['SalePrice'], axis=1)


# 建模、参数调优
kfold = KFold(n_splits=5)
# rand_forest = RandomForestRegressor()
# rand_forest_paras = {"max_depth": [3, 5, 8],
#                      "max_features": ["log2", "sqrt", "auto"],
#                      "bootstrap": [True, False],
#                      "n_estimators": [100, 300]
#                      }
# grid_search = GridSearchCV(rand_forest, rand_forest_paras, cv=kfold, scoring="neg_mean_squared_error", n_jobs=4, verbose=1)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Lasso, ElasticNet
# lasso model kaggle得分： 0.13750 (值越低排名越好)
# lasso = make_pipeline(RobustScaler(), Lasso())
# lasso_paras = {"lasso__alpha": [0.0005, 0.001],
#                "lasso__random_state": [1]
#               }
# {'lasso__alpha': 0.001, 'lasso__random_state': 1}
# grid_search = GridSearchCV(lasso, lasso_paras, cv=kfold, scoring="neg_mean_squared_error", n_jobs=4, verbose=1)

# ElasticNet 得分： 0.13860
# elasticNet = make_pipeline(RobustScaler(), ElasticNet())
# elasticNet_paras = {"elasticnet__alpha": [0.0005, 0.001],
#                "elasticnet__l1_ratio": [0.6, 0.9],
#                "elasticnet__random_state": [1]
#               }
# {'elasticnet__alpha': 0.001, 'elasticnet__l1_ratio': 0.6, 'elasticnet__random_state': 1}
# grid_search = GridSearchCV(elasticNet, elasticNet_paras, cv=kfold, scoring="neg_mean_squared_error", n_jobs=4, verbose=1)

# KernelRidge 得分： 0.35997
# from sklearn.kernel_ridge import KernelRidge
# kernelRidge = KernelRidge()
# kernelRidge_paras = {"alpha": [0.1, 0.6],
#                     "kernel": ['polynomial'],
#                     "degree": [2],
#                     "coef0": [2.5, 5]
#                   }
# grid_search = GridSearchCV(kernelRidge, kernelRidge_paras, cv=kfold, scoring="neg_mean_squared_error", n_jobs=3, verbose=1)

# Gradient Boosting Regression 得分： 0.13045
# from sklearn.ensemble import GradientBoostingRegressor
# gbt = GradientBoostingRegressor()
# gbt_paras = {"n_estimators": [100, 1000],
#              "learning_rate": [0.01, 0.05],
#              "max_depth": [3, 4],
#              "max_features": ["log2", "sqrt", "auto"],
#              "min_samples_leaf": [15],
#              "min_samples_split": [10],
#              "loss": ['huber'],
#              "random_state": [42]
#           }
# {'learning_rate': 0.05, 'loss': 'huber', 'max_depth': 4, 'max_features': 'sqrt', 'min_samples_leaf': 15, 'min_samples_split': 10, 'n_estimators': 1000, 'random_state': 42}
# grid_search = GridSearchCV(gbt, gbt_paras, cv=kfold, scoring="neg_mean_squared_error", n_jobs=3, verbose=1)

# XGBRegressor 得分： 0.13536
# import xgboost as xgb
# xgb = xgb.XGBRegressor()
# xgb_paras = {"colsample_bytree": [0.1, 0.5],
#              "gamma": [0.01, 0.05],
#              "learning_rate": [0.01, 0.05],
#              "max_depth": [3, 5],
#              "n_estimators": [1000, 3000],
#              "reg_alpha": [0.1, 0.5],
#              "reg_lambda": [0.5, 0.9],
#              "random_state": [42]
#           }
# grid_search = GridSearchCV(xgb, xgb_paras, cv=kfold, scoring="neg_mean_squared_error", n_jobs=3, verbose=1)

# LGBMRegressor 得分：0.13133
import lightgbm as lgb
# lgb = lgb.LGBMRegressor()
# lgb_paras = {"objective": ['regression'],
#              "num_leaves": [3, 5],
#              "learning_rate": [0.01, 0.05],
#              "max_bin ": [50, 100],
#              "n_estimators": [1000, 3000],
#              "bagging_fraction ": [0.8],
#              "bagging_freq": [5],
#              "feature_fraction": [0.25],
#              "feature_fraction_seed": [9],
#              "bagging_seed": [9],
#              "min_data_in_leaf ": [6],
#              "min_sum_hessian_in_leaf ": [11]
#           }
# {'bagging_fraction ': 0.8, 'bagging_freq': 5, 'bagging_seed': 9, 'feature_fraction': 0.25, 'feature_fraction_seed': 9, 'learning_rate': 0.01, 'max_bin ': 50, 'min_data_in_leaf ': 6, 'min_sum_hessian_in_leaf ': 11, 'n_estimators': 3000, 'num_leaves': 5, 'objective': 'regression'}
# grid_search = GridSearchCV(lgb, lgb_paras, cv=kfold, scoring="neg_mean_squared_error", n_jobs=3, verbose=1)
# grid_search.fit(X_train, Y_train)
# rand_forest_best = grid_search.best_estimator_
# print(grid_search.best_params_)
# print(grid_search.best_score_)

# AveragingModels 得分：0.12795
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.001, random_state=1))
elasticNet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.001, l1_ratio=0.6, random_state=1))
gbt = GradientBoostingRegressor(learning_rate=0.05, loss='huber', max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, n_estimators=1000, random_state=42)
lgb = lgb.LGBMRegressor(bagging_fraction=0.8, bagging_freq=5, bagging_seed=9, feature_fraction=0.25, feature_fraction_seed=9, learning_rate=0.01, max_bin=50, min_data_in_leaf=6, min_sum_hessian_in_leaf=11, n_estimators=3000, num_leaves=5, objective='regression')
import kaggle_kernel.kaggle_competitions.Housing_Price_Compitition.AveragingModels as avg
# averaged_models = avg.AveragingModels(models=(lasso, elasticNet, gbt, lgb))
# averaged_models.fit(X_train, Y_train)
# predictions = np.expm1(averaged_models.predict(test))

# StackingAveragingModels 得分: 0.12813
stacking_model = avg.StackingModels(base_models=(elasticNet, gbt, lgb), meta_model=lasso)
stacking_model.fit(X_train.values, Y_train)
predictions = np.expm1(stacking_model.predict(test.values))


# Predict and Submit results
# predictions = rand_forest_best.predict(test)
# predictions = np.expm1(rand_forest_best.predict(test))
StackingSubmission = pd.DataFrame({'Id': test_Id, 'SalePrice': predictions})
StackingSubmission.to_csv("./data/stackingmodel_Submission_with_target_log.csv", index=False)
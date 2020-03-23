from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.model_selection import KFold
import numpy as np


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=1)


class StackingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        # 用于保存每个self.base_models下不同Kfold(如等于3)训练出来的模型：结构为[[m1k1,m1k2,m1k3], [m2k1,m2k2,m2k3]...]
        self.base_models_ = [list() for _ in self.base_models]
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                instance.fit(X[train_index], y[train_index])
                self.base_models_[i].append(instance)
                predictions = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = predictions
        self.meta_model.fit(out_of_fold_predictions, y)
        return self

    def predict(self, X):
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        for i, base_model in enumerate(self.base_models_):
            meta_features[:, i] = np.column_stack([model.predict(X) for model in base_model]).mean(axis=1)
        return self.meta_model.predict(meta_features)

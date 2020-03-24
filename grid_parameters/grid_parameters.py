import tensorflow as tf
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X = np.array([
              [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [1, 2], [3, 4], [5, 6], [7, 8],
              [9, 10], [11, 12], [13, 14], [15, 16]
              ])
y = np.array([0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1])

pipe_svc = Pipeline([('scl', StandardScaler()),
                     ('clf', SVC(random_state=1))])
param_grid = [{'clf__drop': [0.5, 0.6, 0.7],
               'clf__learning_rate': [0.01, 0.05, 0.1],
               'clf__l1_regularization_strength': [0.001, 0.005]
            }]

gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=4,
                  n_jobs=1)
gs = gs.fit(X, y)

print(gs.best_score_)
print(gs.best_estimator_)
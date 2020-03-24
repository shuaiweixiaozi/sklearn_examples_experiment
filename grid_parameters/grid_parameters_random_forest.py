import tensorflow as tf
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

X = np.array([
              [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [1, 2], [3, 4], [5, 6], [7, 8],
              [9, 10], [11, 12], [13, 14], [15, 16]
              ])
y = np.array([0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1])

classifier = RandomForestClassifier(verbose=2, n_jobs=1, oob_score=1)

param_grid = {'n_estimators': np.arange(1, 100, 10)}

gs = GridSearchCV(estimator=classifier, param_grid=param_grid)
gs = gs.fit(X, y)

print(gs.best_score_)
print(gs.best_estimator_)
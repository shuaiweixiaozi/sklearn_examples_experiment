from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingCVClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn import model_selection
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools

"""
参考网址： https://rasbt.github.io/mlxtend/user_guide/classifier/StackingCVClassifier/#example-1-simple-stacking-cv-classification
介绍机器学习中stacking技术，并给出相应例子
"""

iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target
RANDOM_SEED = 42


# example1： simple Stacking CV Classification
# 运行结果
# Accuracy: 0.91 (+/- 0.01) [KNN]
# Accuracy: 0.90 (+/- 0.03) [Random Forest]
# Accuracy: 0.92 (+/- 0.03) [Naive Bayes]
# Accuracy: 0.93 (+/- 0.02) [StackingClassifier]
def simple_stacking_cv_classification(X, y):
    clf1 = KNeighborsClassifier(n_neighbors=1)
    clf2 = RandomForestClassifier(random_state=RANDOM_SEED)
    clf3 = GaussianNB()
    lr = LogisticRegression()
    # set 'random_state' to get deterministic result
    sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr, random_state=RANDOM_SEED)
    print('3-fold cross validation:\n')
    for clf, label in zip([clf1, clf2, clf3, sclf], ['KNN', 'Random Forest', 'Naive Bayes', 'StackingClassifier']):
        scores = model_selection.cross_val_score(clf, X, y, cv=3, scoring='accuracy')
        print("Accuracy: %.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    # gs = gridspec.GridSpec(2, 2)
    # fig = plt.figure(figsize=(10, 8))
    # for clf, lab, grd in zip([clf1, clf2, clf3, sclf], ['KNN', 'Random Forest', 'Naive Bayes', 'StackingClassifier'], itertools.product([0, 1], repeat=2)):
    #     clf.fit(X, y)
    #     ax = plt.subplot(gs[grd[0], grd[1]])
    #     fig = plot_decision_regions(X=X, y=y, clf=clf)
    #     plt.title(lab)
    # plt.show()


# example2: using probilities as Meta-Features
# 运行结果
# Accuracy: 0.91 (+/- 0.01) [KNN]
# Accuracy: 0.93 (+/- 0.05) [Random Forest]
# Accuracy: 0.92 (+/- 0.03) [Naive Bayes]
# Accuracy: 0.95 (+/- 0.04) [StackingClassifier]
def simple_stacking_cv_classification_with_probilities_meta_features(X, y):
    clf1 = KNeighborsClassifier(n_neighbors=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    lr = LogisticRegression()

    sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3],
                                use_probas=True,  # 使用概率作为meta class的输入
                                meta_classifier=lr, random_state=RANDOM_SEED)
    for clf, label in zip([clf1, clf2, clf3, sclf], ['KNN', 'Random Forest', 'Naive Bayes', 'StackingClassifier']):
        scores = model_selection.cross_val_score(clf, X, y, cv=3, scoring='accuracy')
        print('Accuracy: %0.2f (+/- %0.2f) [%s]' % (scores.mean(), scores.std(), label))


# simple3_1: stacked cv Classification and GridSearch
# 运行结果
# 0.933 +/- 0.02 {'kneighborsclassifier__n_neighbors': 1, 'meta_classifier__C': 1, 'randomforestclassifier__n_estimators': 10}
# 0.927 +/- 0.02 {'kneighborsclassifier__n_neighbors': 1, 'meta_classifier__C': 1, 'randomforestclassifier__n_estimators': 50}
# 0.907 +/- 0.04 {'kneighborsclassifier__n_neighbors': 1, 'meta_classifier__C': 10, 'randomforestclassifier__n_estimators': 10}
# 0.927 +/- 0.02 {'kneighborsclassifier__n_neighbors': 1, 'meta_classifier__C': 10, 'randomforestclassifier__n_estimators': 50}
# 0.933 +/- 0.03 {'kneighborsclassifier__n_neighbors': 5, 'meta_classifier__C': 1, 'randomforestclassifier__n_estimators': 10}
# 0.933 +/- 0.03 {'kneighborsclassifier__n_neighbors': 5, 'meta_classifier__C': 1, 'randomforestclassifier__n_estimators': 50}
# 0.933 +/- 0.03 {'kneighborsclassifier__n_neighbors': 5, 'meta_classifier__C': 10, 'randomforestclassifier__n_estimators': 10}
# 0.933 +/- 0.03 {'kneighborsclassifier__n_neighbors': 5, 'meta_classifier__C': 10, 'randomforestclassifier__n_estimators': 50}
# Best parameters: {'kneighborsclassifier__n_neighbors': 1, 'meta_classifier__C': 1, 'randomforestclassifier__n_estimators': 10}
# Accuracy: 0.93
def simple_stacking_cv_classification_and_gridsearch(X, y):
    clf1 = KNeighborsClassifier(n_neighbors=1)
    clf2 = RandomForestClassifier(random_state=RANDOM_SEED)
    clf3 = GaussianNB()
    lr = LogisticRegression()

    sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr, random_state=RANDOM_SEED)
    params = {'kneighborsclassifier__n_neighbors': [1, 5],
              'randomforestclassifier__n_estimators': [10, 50],
              'meta_classifier__C': [1, 10]}
    grid = model_selection.GridSearchCV(estimator=sclf, param_grid=params, cv=5, refit=True)
    grid.fit(X, y)
    cv_keys = ('mean_test_score', 'std_test_score', 'params')
    for r, _ in enumerate(grid.cv_results_['mean_test_score']):
        print("%0.3f +/- %0.2f %r" % (grid.cv_results_[cv_keys[0]][r],
                                      grid.cv_results_[cv_keys[1]][r]/2.0,
                                      grid.cv_results_[cv_keys[2]][r]))

    print('Best parameters: %s' % grid.best_params_)
    print('Accuracy: %0.2f' % grid.best_score_)


# simple3_2: stacked cv Classification and GridSearch using a regression algorithm multiple times
# 0.920 +/- 0.06 {'kneighborsclassifier-1__n_neighbors': 1, 'kneighborsclassifier-2__n_neighbors': 1, 'meta_classifier__C': 1, 'randomforestclassifier__n_estimators': 10}
# 0.933 +/- 0.04 {'kneighborsclassifier-1__n_neighbors': 1, 'kneighborsclassifier-2__n_neighbors': 1, 'meta_classifier__C': 1, 'randomforestclassifier__n_estimators': 50}
# 0.920 +/- 0.06 {'kneighborsclassifier-1__n_neighbors': 1, 'kneighborsclassifier-2__n_neighbors': 1, 'meta_classifier__C': 10, 'randomforestclassifier__n_estimators': 10}
# 0.933 +/- 0.04 {'kneighborsclassifier-1__n_neighbors': 1, 'kneighborsclassifier-2__n_neighbors': 1, 'meta_classifier__C': 10, 'randomforestclassifier__n_estimators': 50}
# 0.940 +/- 0.04 {'kneighborsclassifier-1__n_neighbors': 1, 'kneighborsclassifier-2__n_neighbors': 5, 'meta_classifier__C': 1, 'randomforestclassifier__n_estimators': 10}
# 0.947 +/- 0.05 {'kneighborsclassifier-1__n_neighbors': 1, 'kneighborsclassifier-2__n_neighbors': 5, 'meta_classifier__C': 1, 'randomforestclassifier__n_estimators': 50}
# 0.920 +/- 0.05 {'kneighborsclassifier-1__n_neighbors': 1, 'kneighborsclassifier-2__n_neighbors': 5, 'meta_classifier__C': 10, 'randomforestclassifier__n_estimators': 10}
# 0.940 +/- 0.04 {'kneighborsclassifier-1__n_neighbors': 1, 'kneighborsclassifier-2__n_neighbors': 5, 'meta_classifier__C': 10, 'randomforestclassifier__n_estimators': 50}
# 0.940 +/- 0.04 {'kneighborsclassifier-1__n_neighbors': 5, 'kneighborsclassifier-2__n_neighbors': 1, 'meta_classifier__C': 1, 'randomforestclassifier__n_estimators': 10}
# 0.947 +/- 0.05 {'kneighborsclassifier-1__n_neighbors': 5, 'kneighborsclassifier-2__n_neighbors': 1, 'meta_classifier__C': 1, 'randomforestclassifier__n_estimators': 50}
# 0.920 +/- 0.05 {'kneighborsclassifier-1__n_neighbors': 5, 'kneighborsclassifier-2__n_neighbors': 1, 'meta_classifier__C': 10, 'randomforestclassifier__n_estimators': 10}
# 0.940 +/- 0.04 {'kneighborsclassifier-1__n_neighbors': 5, 'kneighborsclassifier-2__n_neighbors': 1, 'meta_classifier__C': 10, 'randomforestclassifier__n_estimators': 50}
# 0.947 +/- 0.05 {'kneighborsclassifier-1__n_neighbors': 5, 'kneighborsclassifier-2__n_neighbors': 5, 'meta_classifier__C': 1, 'randomforestclassifier__n_estimators': 10}
# 0.947 +/- 0.05 {'kneighborsclassifier-1__n_neighbors': 5, 'kneighborsclassifier-2__n_neighbors': 5, 'meta_classifier__C': 1, 'randomforestclassifier__n_estimators': 50}
# 0.940 +/- 0.05 {'kneighborsclassifier-1__n_neighbors': 5, 'kneighborsclassifier-2__n_neighbors': 5, 'meta_classifier__C': 10, 'randomforestclassifier__n_estimators': 10}
# 0.940 +/- 0.05 {'kneighborsclassifier-1__n_neighbors': 5, 'kneighborsclassifier-2__n_neighbors': 5, 'meta_classifier__C': 10, 'randomforestclassifier__n_estimators': 50}
# Best paras: {'kneighborsclassifier-1__n_neighbors': 1, 'kneighborsclassifier-2__n_neighbors': 5, 'meta_classifier__C': 1, 'randomforestclassifier__n_estimators': 50}
# Best scores: 0.947
def stacking_clf_and_gridsearch_and_same_algo_multi_times(X, y):
    clf1 = KNeighborsClassifier(n_neighbors=1)
    clf2 = RandomForestClassifier(random_state=RANDOM_SEED)
    # clf3 = GaussianNB
    lr = LogisticRegression()

    # using a regression algorithm multiple times
    sclf = StackingCVClassifier(classifiers=[clf1, clf1, clf2], meta_classifier=lr, random_state=RANDOM_SEED)
    params = {'kneighborsclassifier-1__n_neighbors': [1, 5],  # add an additional number suffix in the parameter grid
              'kneighborsclassifier-2__n_neighbors': [1, 5],  # add an additional number suffix in the parameter grid
              'randomforestclassifier__n_estimators': [10, 50],
              'meta_classifier__C': [1, 10]}
    grid = model_selection.GridSearchCV(estimator=sclf, param_grid=params, cv=5, refit=True)
    grid.fit(X, y)
    cv_key = ('mean_test_score', 'std_test_score', 'params')
    for r, _ in enumerate(grid.cv_results_['mean_test_score']):
        print('%0.3f +/- %0.2f %r' % (grid.cv_results_[cv_key[0]][r],
                                      grid.cv_results_[cv_key[1]][r],
                                      grid.cv_results_[cv_key[2]][r]))
    print('Best paras: %s' % grid.best_params_)
    print('Best scores: %0.3f' % grid.best_score_)


if __name__ == '__main__':
    # simple_stacking_cv_classification(X, y)
    # simple_stacking_cv_classification_with_probilities_meta_features(X, y)
    # simple_stacking_cv_classification_and_gridsearch(X, y)
    stacking_clf_and_gridsearch_and_same_algo_multi_times(X, y)

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


# simple Stacking CV Classification
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
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(figsize=(10, 8))
    for clf, lab, grd in zip([clf1, clf2, clf3, sclf], ['KNN', 'Random Forest', 'Naive Bayes', 'StackingClassifier'], itertools.product([0, 1], repeat=2)):
        clf.fit(X, y)
        ax = plt.subplot(gs[grd[0], grd[1]])
        fig = plot_decision_regions(X=X, y=y, clf=clf)
        plt.title(lab)
    plt.show()


if __name__ == '__main__':
    simple_stacking_cv_classification(X, y)

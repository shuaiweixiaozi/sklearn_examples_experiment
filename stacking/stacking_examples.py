from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingCVClassifier
from sklearn import model_selection

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


if __name__ == '__main__':
    simple_stacking_cv_classification(X, y)

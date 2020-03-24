from sklearn.utils import check_random_state
from sklearn import datasets
import numpy as np
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, ShuffleSplit

rnd = check_random_state(1)

# set up dataset
n_samples = 100
n_features = 300

# l1 data (only 5 informative features)
X_1, y_1 = datasets.make_classification(n_samples=n_samples, n_features=n_features, n_informative=5)

# l2 data: non sparse, but less features
y_2 = np.sign(0.5 - rnd.randn(n_samples))
X_2 = rnd.randn(n_samples, n_features // 5) + y_2[:, np.newaxis]
X_2 += 5 * rnd.randn(n_samples, n_features // 5)

clf_sets = [(LinearSVC(penalty='l1', loss='squared_hinge', dual=False, tol=1e-3), np.logspace(-2.3, -1.3, 10), X_1, y_1),
            (LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=1e-4), np.logspace(-4.5, -2, 10), X_2, y_2)]
colors = ['navy', 'cyan', 'darkorange']
lw = 2

for clf, cs, X, y in clf_sets:
    # set up the plot for each regressor
    fig, axes = plt.subplots(nrows=2, sharey=True, figsize=(9, 10))
    for k, train_size in enumerate(np.linspace(0.3, 0.7, 3)[::-1]):
        param_grid = dict(C=cs)
        # to get nice curve, we need a large number of iterations to reduce the variance
        grid = GridSearchCV(clf, refit=False, param_grid=param_grid,
                    cv=ShuffleSplit(train_size=train_size, test_size=0.3, n_splits=250, random_state=1))
        grid.fit(X, y)
        scores = grid.cv_results_['mean_test_score']
        scales = [(1, 'No scaling'), ((n_samples * train_size), '1/n_samples'),]
        for ax, (scaler, name) in zip(axes, scales):
            ax.set_xlabel('C')
            ax.set_ylabel('CV Score')
            grid_cs = cs * float(scaler)  # scale the C's
            ax.semilogx(grid_cs, scores, label="fraction %.2f" % train_size, color=colors[k], lw=lw)
            ax.set_title('scaling=%s, penalty=%s, loss=%s' % (name, clf.penalty, clf.loss))
    plt.legend(loc="best")
plt.show()
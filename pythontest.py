import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

# Settings
n_repeat = 50       # Number of iterations for computing expectations
n_train = 50        # Size of the training set
n_test = 1000       # Size of the test set
noise = 0.1         # Standard deviation of the noise
np.random.seed(0)

# Change this for exploring the bias-variance decomposition of other
# estimators. This should work well for estimators with high variance (e.g.,
# decision trees or KNN), but poorly for estimators with low variance (e.g.,
# linear models).
estimators = [("Tree", DecisionTreeRegressor()),
              ("Bagging(Tree)", BaggingRegressor(DecisionTreeRegressor()))]

n_estimators = len(estimators)


# Generate data
def f(x):
    x = x.ravel()

    return np.exp(-x ** 2) + 1.5 * np.exp(-(x - 2) ** 2)


def generate(n_samples, noise, n_repeat=1):
    X = np.random.random(n_samples) * 10 - 5
    X = np.sort(X)

    if n_repeat == 1:
        y = f(X) + np.random.normal(0, noise, n_samples)
    else:
        y = np.zeros((n_samples, n_repeat))
        for i in range(n_repeat):
            y[:, i] = f(X) + np.random.normal(0, noise, n_samples)
    X = X.reshape((n_samples, 1))
    return X, y


X_trains = []
y_trains = []
for i in range(n_repeat):
    X, y = generate(n_train, noise)
    X_trains.append(X)
    y_trains.append(y)

X_test, y_test = generate(n_test, noise, n_repeat)

clf = DecisionTreeRegressor()
y_predicts = np.zeros((n_test, n_repeat))
for i in range(n_repeat):
    clf.fit(X_trains[i], y_trains[i])
    y_predicts[:, i] = clf.predict(X_test)


y_error = np.zeros(n_test)
for i in range(n_repeat):
    for j in range(n_repeat):
        y_error += (y_test[:, j] - y_predicts[:, i]) ** 2


y_error /= (n_repeat * n_repeat)
y_noise = np.var(y_test, axis=1)
y_bias = (f(X_test) - np.mean(y_predicts, axis=1)) ** 2
y_variance = np.var(y_predicts, axis=1)
print('%s: %.4f (error)= %.4f (bias^2) + %.4f (var) + %.4f (noise)'
      % ("Tree", np.mean(y_error), np.mean(y_bias), np.mean(y_variance), np.mean(y_noise)))

plt.subplot(2, 2, 1)
plt.plot(X_test, f(X_test), 'b', label="$f(x)$")
plt.plot(X_trains[0], y_trains[0], '.b', label='LS ~ $y = f(x) + noise$')

for i in range(n_repeat):
    if i == 0:
        plt.plot(X_test, y_predicts[:, i], 'r', label='$\^y(x)$')
    else:
        plt.plot(X_test, y_predicts[:, i], 'r', alpha=0.05)

plt.plot(X_test, np.mean(y_predicts, axis=1), "c", label=r"\mathbb{E}_{LS}\^y(x)$")
plt.xlim([-5, 5])
plt.title("Tree")

plt.legend(loc=(1.1, 0.5))

plt.subplot(2, 2, 3)
plt.plot(X_test, y_error, 'r', label='$error(x)$')
plt.plot(X_test, y_bias, 'b', label='$bias^2(x)$')
plt.plot(X_test, y_variance, 'g', label='$variance(x)$')
plt.plot(X_test, y_noise, 'c', label='$noise(x)$')

plt.xlim([-5, 5])
plt.ylim([0, 0.1])
plt.legend(loc=(1.1, 0.5))

plt.subplots_adjust(right=0.75)
plt.show()


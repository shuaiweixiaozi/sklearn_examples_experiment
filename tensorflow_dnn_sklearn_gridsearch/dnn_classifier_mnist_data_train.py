from dnn_classifier import DNNClassifier
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import accuracy_score
import tensorflow as tf
from sklearn.model_selection import GridSearchCV

mnist = input_data.read_data_sets("MNIST_data")

X_train = mnist.train.images
y_train = mnist.train.labels

X_validation = mnist.validation.images
y_validation = mnist.validation.labels

X_test = mnist.test.images
y_test = mnist.test.labels

dnn = DNNClassifier(tensorboard_logdir="tensorboard_stats", random_state=42)
# dnn.fit(X_train, y_train, 100, X_validation, y_validation)
# y_pred = dnn.predict(X_test)
# print("Accuracy on the test set: {:.2f}".format(accuracy_score(y_test,y_pred) * 100))

# use randomized search to find the best hyperparameters
parameter_distributions = {
    'n_hidden_layers': [3, 4, 5],
    'n_neurons': [40, 50, 100],
    'batch_size' : [64, 128],
    'learning_rate': [0.01, 0.005],
    'activation': [tf.nn.elu, tf.nn.relu],
    'max_checks_without_progress': [20, 30],
    'batch_norm_momentum': [None, 0.9],
    'dropout_rate': [None, 0.5]
}

random_search = GridSearchCV(dnn, parameter_distributions, scoring='accuracy', verbose=2)
random_search.fit(X_train, y_train)
best_params = random_search.best_params_
best_estimator = random_search.best_estimator_
mnist_predictions = best_estimator.predict(X_test)
print("Accuracy on the test set: {:.2f}".format(accuracy_score(y_test, mnist_predictions) * 100))
best_estimator.save("models/grid_best_model")

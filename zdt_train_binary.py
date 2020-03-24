# -*- coding: utf-8 -*-
"""

A classification model for prediction of brokers on LongHuBang

@author: admin
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import tensorflow as tf
import bage.fe.utils as utils

tf.logging.set_verbosity(tf.logging.INFO)


FEATURES = ['change3', 'change5', 'change10', 'change20', 'change60',
           'offset10', 'offset120', 'offset20', 'offset30', 'offset5', 'offset60', 'volmacd', 'weight']


FEATURES_NUMERIC = ['change3', 'change5', 'change10', 'change20', 'change60',
           'offset10', 'offset120', 'offset20', 'offset30', 'offset5', 'offset60', 'volmacd']

# 炸板标志：0-炸板，1-没有炸板
LABEL = "zdt_x"


def get_weight(count_zdt1, count_zdt0):
    def get_weight_func(zdt):
        if zdt == 0:
            weight = 10000.0/count_zdt0
        else:
            weight = 10000.0/count_zdt1
        return weight
    return get_weight_func


def get_input_fn(data_set, num_epochs=None, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
        y=pd.Series(data_set[LABEL].values),
        num_epochs=num_epochs,
        shuffle=shuffle)


def load_data():
    df = utils.load_df_from_csv("zdt_train_samples")
    df.dropna(inplace=True)
    df[LABEL] = df[LABEL].apply(lambda x: round(x))
    print(df[LABEL].unique())

    print(df.head(10))
    df_zdt_1 = df[df[LABEL] == 1]
    df_zdt_0 = df[df[LABEL] == 0]
    zdt_1 = len(df_zdt_1)
    zdt_0 = len(df_zdt_0)
    df['weight'] = df[LABEL].apply(get_weight(zdt_1, zdt_0))

    df_norm = df[FEATURES_NUMERIC]
    # z-score normalization
    mean = df_norm.mean()
    std = df_norm.std()
    # diff all samples in Dataframe with the sample s
    df_norm = (df_norm - mean) / std
    df[FEATURES_NUMERIC] = df_norm

    #df = df.sort_values(by=['date'], ascending=True)
    count = len(df)
    prediction_set = df.tail(100)
    train_set = df.head(count - 100)
    count = count - 100
    training_set = train_set.head(round(count*0.8))
    test_set = train_set.tail(round(count*0.2))
    training_set = training_set.sample(frac=1).reset_index(drop=True)

    print(prediction_set)
    return training_set, test_set, prediction_set


def main(unused_argv):
    # Load datasets
    training_data, test_data, prediction_data = load_data()

    # Feature cols
    feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES_NUMERIC]
    weight_column = tf.feature_column.numeric_column('weight')

    # Build 2 layer fully connected DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_cols,
                                            hidden_units=[128, 128],
                                            n_classes=2,
                                            #dropout=0.5,
                                            weight_column=weight_column,
                                            optimizer=tf.train.ProximalAdagradOptimizer(
                                                learning_rate=0.05,
                                                l1_regularization_strength=0.001
                                            ),
                                            model_dir="/tmp/zdt_model_9")

    print('data loaded!')
    print(len(training_data))
    print(len(test_data))
    print(len(prediction_data))

    # Train
    classifier.train(input_fn=get_input_fn(training_data), steps=200000)

    # Evaluate loss over one epoch of test_set.
    ev = classifier.evaluate(input_fn=get_input_fn(test_data, num_epochs=1, shuffle=False))
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**ev))

    # Print out predictions over a slice of prediction_set.
    y = classifier.predict(
        input_fn=get_input_fn(prediction_data, num_epochs=1, shuffle=False))

    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
    expected = prediction_data[LABEL].values
    total = len(prediction_data)
    correct = 0
    for predict, expect in zip(y, expected):
        print(predict)
        class_id = predict['class_ids'][0]
        probability = predict['probabilities'][class_id]
        print(template.format(class_id, 100 * probability, expect))
        if expect == class_id:
            correct += 1
    print('\nPrediction set accuracy: {:.1f}%\n'.format(100*correct/float(total)))


if __name__ == "__main__":
    tf.app.run()
    #load_data()

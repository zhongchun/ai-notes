#! /usr/bin/env python
# -*- coding: utf-8 -*ﬁ-
"""
 @FileName: 3_classification
 @Desc:  
 @Author: yuzhongchun
 @Date: 2019-02-19 21:56
 @Last Modified by: yuzhongchun
 @Last Modified time: 2019-02-19 21:56
"""
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals, absolute_import

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import pandas as pd
import sys
from common.util import *

# to make output stable across runs
np.random.seed(42)

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# basic info
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "classification"


def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]


# Plot some digits from the dataset
def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap=mpl.cm.binary,
               interpolation="nearest")
    plt.axis("off")


def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row: (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    print_line(row_images[0].shape, name='row_images[0].shape')
    print_line(image.shape, name='image.shape')
    plt.imshow(image, cmap=mpl.cm.binary, **options)
    plt.axis("off")


# Get the data
mnist = fetch_openml('mnist_784', version=1, cache=True)
mnist.target = mnist.target.astype(np.int8)  # fetch_openml() returns targets as strings
sort_by_target(mnist)  # fetch_openml() returns an unsorted dataset

# Take a look at one digit from the dataset
print_line(mnist, name='mnist')
print_line(mnist["data"], name='mnist["data"]')
print_line(mnist["target"], name='mnist["target"]')

X, y = mnist["data"], mnist["target"]
print_line(X.shape, name='X.shape')
print_line(y.shape, name='y.shape')

some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=mpl.cm.binary, interpolation="nearest")
plt.axis("off")
save_fig("some_digit_plot", PROJECT_ROOT_DIR, CHAPTER_ID)
plt.show()

print_line(y[36000], 'y[36000]')

plt.figure(figsize=(9, 9))
example_images = np.r_[X[:12000:600], X[13000:30600:600], X[30600:60000:590]]
plot_digits(example_images, images_per_row=10)
save_fig("more_digits_plot", PROJECT_ROOT_DIR, CHAPTER_ID)
plt.show()

y_data_frame = pd.DataFrame(y)
print_line(y_data_frame[0].value_counts(), 'y_data_frame')

print_line(X[:12000:600].shape, 'X[:12000:600],shape')
print_line(X[13000:30600:600].shape, 'X[13000:30600:600].shape')
print_line(X[30600:60000:590].shape, 'X[30600:60000:590],shape')
print_line(example_images, 'example_images')
print_line(example_images.shape, 'example_images.shape')

# Get a train and test dataset
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# Train a binary classifier
# True for all 5s, False for all other digits
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

print_line(y_train_5.shape)
print_line(y_train_5)

# Test some model
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)
sgd_clf.fit(X_train, y_train_5)

print_line(sgd_clf)
print_line(sgd_clf.predict([some_digit]))

from sklearn.model_selection import cross_val_score

print_line(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = (y_train_5[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_5[test_index])

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))

from sklearn.base import BaseEstimator


class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


never_5_clf = Never5Classifier()
print_line(cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy"))

from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

from sklearn.metrics import confusion_matrix

print_line(confusion_matrix(y_train_5, y_train_pred))

y_train_perfect_predictions = y_train_5
print_line(confusion_matrix(y_train_5, y_train_perfect_predictions))

from sklearn.metrics import precision_score, recall_score, f1_score

print_line(precision_score(y_train_5, y_train_pred))
print_line(recall_score(y_train_5, y_train_pred))
print_line(f1_score(y_train_5, y_train_pred))

y_scores = sgd_clf.decision_function([some_digit])
print_line(y_scores)
threshold = 0
y_some_digit_pred = (y_scores > threshold)
print_line(y_some_digit_pred)

threshold = 200000
y_some_digit_pred = (y_scores > threshold)
print_line(y_some_digit_pred)

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
print_line(y_scores.shape)

# hack to work around issue #9589 in Scikit-Learn 0.19.0
if y_scores.ndim == 2:
    y_scores = y_scores[:, 1]

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])


plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.xlim([-700000, 700000])
save_fig("precision_recall_vs_threshold_plot", PROJECT_ROOT_DIR, CHAPTER_ID)
plt.show()

(y_train_pred == (y_scores > 0)).all()
y_train_pred_90 = (y_scores > 70000)
print_line(precision_score(y_train_5, y_train_pred_90))
print_line(recall_score(y_train_5, y_train_pred_90))


def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])


plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
save_fig("precision_vs_recall_plot", PROJECT_ROOT_DIR, CHAPTER_ID)
plt.show()

# ROC
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)


plt.figure(figsize=(8, 6))
plot_roc_curve(fpr, tpr)
save_fig("roc_curve_plot", PROJECT_ROOT_DIR, CHAPTER_ID)
plt.show()

from sklearn.metrics import roc_auc_score

print_line(roc_auc_score(y_train_5, y_scores))

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=10, random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
y_scores_forest = y_probas_forest[:, 1]  # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right", fontsize=16)
save_fig("roc_curve_comparison_plot", PROJECT_ROOT_DIR, CHAPTER_ID)
plt.show()

print_line(roc_auc_score(y_train_5, y_scores_forest))
y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3)
print_line(precision_score(y_train_5, y_train_pred_forest))
print_line(recall_score(y_train_5, y_train_pred_forest))

# Multiclass classification
sgd_clf.fit(X_train, y_train)
print_line(sgd_clf.predict([some_digit]))

some_digit_scores = sgd_clf.decision_function([some_digit])
max_index = np.argmax(some_digit_scores)
print_line(some_digit_scores, name='some_digit_scores')
print_line(max_index, name='max_index')
print_line(sgd_clf.classes_, name='sgd_clf.classes_')
print_line(sgd_clf.classes_[5], name='sgd_clf.classes_[5]')

from sklearn.multiclass import OneVsOneClassifier

ovo_clf = OneVsOneClassifier(SGDClassifier(max_iter=5, tol=-np.infty, random_state=42))
ovo_clf.fit(X_train, y_train)
print_line(ovo_clf.predict([some_digit]), name='predicted som_digit')
print_line(len(ovo_clf.estimators_), name='len(ovo_clf.estimators_)')

forest_clf.fit(X_train, y_train)
print_line(forest_clf.predict([some_digit]), name='predicted som_digit')
print_line(forest_clf.predict_proba([some_digit]), name='forest predict probabilities')

sgd_scores = cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
print_line(sgd_scores, name='sgd_scores')

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
sgd_scaled_scores = cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
print_line(sgd_scaled_scores, name='sgd_scaled_scores')

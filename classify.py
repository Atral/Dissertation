import pandas as pd
import time
from timeit import default_timer as timer
from datetime import timedelta
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.metrics import f1_score, make_scorer, confusion_matrix, plot_confusion_matrix, classification_report
X = []
y = []

with open("processed.txt", "r") as f:
    print("Reading processed data...")
    for line in f:
        xy = line.split("|")
        X.append(eval(xy[0]))
        y.append(xy[1])

print("Vectorising...")
vec = DictVectorizer(sparse=False)
X = vec.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# Classifying
clf = MultinomialNB().fit(X_train, y_train)
# clf = svm.SVC()
print("Fitting classifier...")
clf.fit(X_train, y_train)

with open("test.txt", "w") as f:
    # print("Scoring...")
    # print(clf.score(X_test, y_test))
    print("Predicting...")
    y_pred = clf.predict(X_test)
    inv = vec.inverse_transform(X_train)
    rev_input = []

    for i, v in enumerate(inv):
        inv = str(vec.inverse_transform(X_train)[i].keys()).split("=")
        t = inv[1].split(",")[0][:-1]
        rev_input.append(t)

    # print("calculating f1..")
    # print(f1_score(y_test, y_pred, average="micro"))
    # print(classification_report(y_test, y_pred))
    changed = 0
    correct = 0

    print("Performing secondary evaluation...")
    for orig, inpt, prediction, label in zip(rev_input, X_test, y_pred, y_test):
        #if prediction != label:
           #print(inpt, 'has been classified as ', prediction, 'and should be ', label)
        # print(inpt, label)
        if orig != label:
            changed += 1
            if label == prediction:
                correct += 1

    print(correct/changed)

            # plot_confusion_matrix(clf, X_test, y_test)
    # plt.show()
    # inv = vec.inverse_transform(X_test)

#     inv = vec.get_feature_names()
#     print(len(inv))
#     answers = []
#     print(inv)
#     for i in range(len(inv)-1):
#         answers.append(vec.get_feature_names()[i].split("=")[1])
#
#     print(len(answers))
#     print(len(y_test))
#
#     for i in range(len(y_test)):
#         tok = inv[i].get("token")
#         w = str(tok) + "->" + str(prediction[i]) + " :: " + str(y_test[i]) + "\n"
#
#         if str(tok) != str(y_test[i]):
#             f.write(w)
#             changed += 1
#             if prediction[i] == y_test[i]:
#                 correct += 1
#
# print(correct/changed)

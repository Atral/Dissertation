import nltk
import pandas as pd
import time
from timeit import default_timer as timer
from datetime import timedelta
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import re
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer, confusion_matrix, plot_confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier

X = []
y = []


def plotSVC(title):
  # create a mesh to plot in
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  h = (x_max / x_min)/100
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
  np.arange(y_min, y_max, h))
  plt.subplot(1, 1, 1)
  Z = svm.SVC.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)
  plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
  plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
  plt.xlabel('Original')
  plt.ylabel('Correction')
  plt.xlim(xx.min(), xx.max())
  plt.title(title)
  plt.show()


with open("processed_tid-prnt-tok", "r") as f:
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
kernels = ['linear', 'rbf', 'poly']
cs = [0.2, 0.3, 0.4]
clfs = [MultinomialNB().fit(X_train, y_train), svm.SVC(kernel='linear', C=0.5),
        RandomForestClassifier(random_state=1), DecisionTreeClassifier(), LogisticRegression()]

for c in clfs:
    clf = c
    print("Fitting classifier...")

    clf.fit(X_train, y_train)

    with open("classifier-test.txt", "a") as f:
        f.write("SVC WITH C=" + str(c) + " AND DEFAULT PARAMS: ")
        print("Scoring...")
        score = clf.score(X_test, y_test)
        print("Overall: ", score)
        f.write("Overall = " + str(score))
        print("Predicting...")
        y_pred = clf.predict(X_test)
        # inv = vec.inverse_transform(X_train)

        rev_input = []

        for i, v in enumerate(X_test):
            a = str(vec.inverse_transform(X_test[i][None, :]))
            b = (re.findall("\{(.*?)\}", a))[0]
            c = b.split(",")[1].split(":")[0][2:-1]

            if len(c.split("=")) >= 2:
                d = c.split("=")[1]
                rev_input.append(d)

        # print("calculating f1..")
        # print(f1_score(y_test, y_pred, average="micro"))
        # print(classification_report(y_test, y_pred))
        changed = 0
        correct = 0
        incorrectly_changed = 0
        incorrect_unchanged = 0

        print("Performing secondary evaluation...")
        for inpt, prediction, label in zip(rev_input, y_pred, y_test):
            label = label.replace('\n', '')
            prediction = prediction.replace('\n', '')
            with open("output.txt", "a") as f2:
                f2.write(str(inpt) + " => " + str(prediction) + " | " + str(label) + '\n')

            if inpt != label:
                #f.write(":" + str(inpt) + ": should be corrected to :" + str(label) + ":\n")
                changed += 1
                if label == prediction:
                    #f.write("=> correctly changed :" + str(inpt) + ":" + " to :" + str(prediction) + ":\n")
                    #f.write("\n")
                    correct += 1
                #else:
                    #f.write("=> incorrectly changed :" + str(inpt) + ":" + " to :" + str(prediction) + ":\n")
                    #f.write("\n")

            if (prediction != label) & (inpt == label):
                incorrectly_changed += 1

            if (inpt != label) & (prediction == inpt):
                incorrect_unchanged += 1

        print("Changed: ", round(correct/changed, 4))
        f.write(" Changed = " + str(round(correct/changed, 4)) + ". ")
        f.write(str(round((changed - correct) / len(rev_input), 2)) + " given wrong correction. " +
                str(round(incorrectly_changed / len(rev_input), 2)) + " changed when they shouldn't have been and " +
                str(incorrect_unchanged / len(rev_input)) + " not changed when they should have been.\n")

        # plot_confusion_matrix(clf, X_test, y_test)
        # plt.show()
        # inv = vec.inverse_transform(X_test)

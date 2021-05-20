import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer, confusion_matrix, plot_confusion_matrix, classification_report, \
    accuracy_score
from sklearn.tree import DecisionTreeClassifier
import statistics


def read_data(filename, X_array, y_array):
    with open(filename, "r") as f:
        print("Reading processed data...")

        for line in f:
            xy = line.split("|")
            X_array.append(eval(xy[0]))
            y_array.append(xy[1])


X_train = []
y_train = []
X_test = []
y_test = []

TRAINPATH = "top5_train.txt"
TESTPATH = "top5_test.txt"
OUTPUT = "topwords.csv"

read_data(TRAINPATH, X_train, y_train)
read_data(TESTPATH, X_test, y_test)

# Vectorising imported data
print("Vectorising...")
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

# SVM params
kernels = ['linear', 'rbf', 'poly']
cs = [0.05, 0.2, 0.3, 0.4]

#NB params
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 5, 10]

# RF Params
estimators = [100, 300, 500, 750, 800, 1200]
gammas = [1,0.1,0.01,0.001]

# LR params
max_depths = np.linspace(33, 60, 28, endpoint=True)
solvers = ['newton-cg', 'lbfgs', 'liblinear']
clfs = [MultinomialNB(), svm.SVC(),
        RandomForestClassifier(random_state=1), DecisionTreeClassifier(), LogisticRegression()]

# DT params
min_samples_splits = [2, 5, 10, 50, 100]
c2s = [100, 10, 1.0, 0.1, 0.01]

optims = [LogisticRegression(solver='liblinear'), svm.SVC(kernel='linear'), MultinomialNB(alpha=0.5),
          RandomForestClassifier(random_state=1, n_estimators = 500), DecisionTreeClassifier(max_depth=45, min_samples_split=5)]

with open(OUTPUT, "a") as f:
    # f.write("Classifier, Overall, Changed, WrongCorrection, IncorrectlyChanged, IncorrectlyUnchanged\n")

    for v in range(1):
        clf =  RandomForestClassifier(random_state=1, n_estimators = 500)
        print("Fitting classifier...")
        clf.fit(X_train, y_train)
        f.write(str(10) + ",")

        print("Scoring...")
        t_score = clf.score(X_train, y_train)

        print("Predicting...")
        y_pred = clf.predict(X_test)
        # inv = vec.inverse_transform(X_train)

        rev_input = []

        for i, v in enumerate(X_test):
            a = str(vec.inverse_transform(X_test[i][None, :]))
            res = a.split('token=', maxsplit=1)[-1] \
                .split(maxsplit=1)[0][:-2]

            if "=" in res:
                rev_input.append("")
            else:
                rev_input.append(res)
                # print(res, " extraced from ", a)

        # print(classification_report(y_test, y_pred))
        changed = 0
        correct = 0
        incorrectly_changed = 0
        incorrect_unchanged = 0
        incorrect = 0

        print("Performing secondary evaluation...")
        for inpt, prediction, label in zip(rev_input, y_pred, y_test):
            label = label.rstrip()
            prediction = prediction.rstrip()
            if prediction != label:
                incorrect += 1

            if inpt != label:
                changed += 1
                with open("output.txt", "a") as f2:
                    f2.write(inpt + " => " + prediction + " | " + label + '\n')
                    # print("--->", label)

                if label == prediction:
                    with open("output.txt", "a") as f2:
                        f2.write("****************" + "\n")
                    correct += 1

            if (prediction != label) & (inpt == label):
                incorrectly_changed += 1

            if (inpt != label) & (prediction == inpt):
                incorrect_unchanged += 1

        # print(classification_report(y_test, y_pred))
        cv_score = round(accuracy_score(y_test, y_pred)*100, 4)
        print("Overall score: ", cv_score, "%")
        print("Changed: ", correct , " correct of ", changed, " making ", round(correct / changed, 4)*100, "%")
        f.write(str(cv_score) + ",")
        f.write(str(round(correct / changed, 4)*100) + ",")
        f.write(str(round((changed - correct) / incorrect, 4)) + "," +
                str(round(incorrectly_changed / incorrect, 4)) + "," +
                str(round(incorrect_unchanged / incorrect, 4)) + "\n")

    # plot_confusion_matrix(clf, X_test, y_test)
    # plt.show()

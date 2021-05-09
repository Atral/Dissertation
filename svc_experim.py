import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer, confusion_matrix, plot_confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier


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

TRAINPATH = "base_train-shuffled"
TESTPATH = "base_test-shuffled"

read_data(TRAINPATH, X_train, y_train)
read_data(TESTPATH, X_test, y_test)

print("Vectorising...")
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Classifying
kernels = ['linear', 'rbf', 'poly']
cs = [0.1, 0.5, 1, 3, 5, 20, 50, 100]
alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3]
estimators = [100, 300, 500, 750, 800, 1200]
clfs = [MultinomialNB(), svm.SVC(),
        RandomForestClassifier(random_state=1), DecisionTreeClassifier(), LogisticRegression()]

with open("classifier-selection.csv", "a") as f:
    f.write("Classifier, Overall, Changed, WrongCorrection, IncorrectlyChanged, IncorrectlyUnchanged\n")

    for c in clfs:
        clf = c
        print("Fitting classifier...")
        clf.fit(X_train, y_train)
        f.write(str(c) + ",")

        print("Scoring...")
        score = clf.score(X_test, y_test)

        print("Overall: ", score)
        f.write(str(round(score, 4)) + ",")

        print("Predicting...")
        y_pred = clf.predict(X_test)
        # inv = vec.inverse_transform(X_train)

        rev_input = []

        for i, v in enumerate(X_test):
            a = str(vec.inverse_transform(X_test[i][None, :]))
            b = (re.findall("\{(.*?)\}", a))[0]
            if len(b.split(",")) > 1:
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
        incorrect = 0

        print("Performing secondary evaluation...")
        for inpt, prediction, label in zip(rev_input, y_pred, y_test):
            label = label.replace('\n', '')
            prediction = prediction.replace('\n', '')
            with open("output.txt", "a") as f2:
                f2.write(str(inpt) + " => " + str(prediction) + " | " + str(label) + '\n')
            if prediction != label:
                incorrect += 1
            if inpt != label:
                # f.write(":" + str(inpt) + ": should be corrected to :" + str(label) + ":\n")
                changed += 1
                if label == prediction:
                    # f.write("=> correctly changed :" + str(inpt) + ":" + " to :" + str(prediction) + ":\n")
                    # f.write("\n")
                    correct += 1
                # else:
                # f.write("=> incorrectly changed :" + str(inpt) + ":" + " to :" + str(prediction) + ":\n")
                # f.write("\n")

            if (prediction != label) & (inpt == label):
                incorrectly_changed += 1

            if (inpt != label) & (prediction == inpt):
                incorrect_unchanged += 1

        print("Changed: ", round(correct / changed, 4))
        f.write(str(round(correct / changed, 4)) + ",")
        f.write(str(round((changed - correct) / incorrect, 4)) + "," +
                str(round(incorrectly_changed / incorrect, 4)) + "," +
                str(round(incorrect_unchanged / incorrect, 4)) + "\n")

    # plot_confusion_matrix(clf, X_test, y_test)
    # plt.show()
    # inv = vec.inverse_transform(X_test)

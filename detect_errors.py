import re
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


data = []
columns = ["X", "CORRECTION", "CHANGED"]

with open("processed_0.5.txt", "r") as f:
    print("Reading processed data...")
    for line in f:
        xy = line.split("|")
        tok = eval(xy[0]).get("token")
        corr = xy[1].rstrip()

        if str(tok) == str(corr):
            data.append([eval(xy[0]), corr, 0])
        else:
            data.append([eval(xy[0]), corr, 1])

df = pd.DataFrame(data, columns=columns)
X = df["X"].to_list()
y = df["CHANGED"].to_list()
vec = DictVectorizer(sparse=False)
X = vec.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=1)
clfs = [MultinomialNB().fit(X_train, y_train), svm.SVC(), svm.SVC(kernel="linear", C=0.025), svm.SVC(gamma=2, C=1),
        RandomForestClassifier(max_depth=5, n_estimators=50, max_features=1, n_jobs=-1, random_state=1),
        DecisionTreeClassifier(max_depth=20), LogisticRegression(n_jobs=1, C=1e5, max_iter=2000)]
print("Fitting classifier...")


clf = clfs[5]
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
# print(f1_score(y_test, y_pred, average="micro"))
print(classification_report(y_test, y_pred))

score = clf.score(X_test, y_test)

rev_input = []

for i, v in enumerate(X_test):
    a = str(vec.inverse_transform(X_test[i][None, :]))
    b = (re.findall("\{(.*?)\}", a))[0]
    c = b.split(",")[1].split(":")[0][1:-1]

    if len(c.split("=")) >= 2:
        d = c.split("=")[1]
        rev_input.append(d)


# for X, y, pred in zip(rev_input, y_test, y_pred):
#     print(X, y, pred)

print(score)
plot_confusion_matrix(clf, X_test, y_test)
plt.show()
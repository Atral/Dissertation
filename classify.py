from sklearn.pipeline import Pipeline, FeatureUnion

from load_data import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm

SGML_PATH = "DATA/release3_2_PARSED/data/conll14st-preprocessed.conll.ann"
DATA_DIR = "DATA/release3_2_PARSED/data/conll14st-preprocessed.conll"
cols = ["NID", "PID", "SID", "TOKENID", "TOKEN", "POS", "DPHEAD", "DPREL", "SYNT", "CORRECTION"]

df = pd.read_table(DATA_DIR, encoding="ISO-8859-1", names=cols)
df["CORRECTION"] = df["TOKEN"]

# paragraphs = find_paragraphs(SGML_PATH)
mistakes = find_mistakes(SGML_PATH)
prep_errors = filter_by_mistake_type(mistakes, "Prep")


for error in prep_errors:
    conditions = [df.NID == eval(error["nid"]), df.PID == eval(error["pid"]), df.SID == eval(error["sid"]), df.TOKENID >= eval(error["start_token"]), df.TOKENID < eval(error["end_token"])]
    mask = (df.NID == eval(error["nid"]))
    mask &= (df.PID == eval(error["pid"]))
    mask &= (df.SID == eval(error["sid"]))
    mask &= (df.TOKENID >= eval(error["start_token"]))
    mask &= (df.TOKENID < eval(error["end_token"]))
    df.loc[mask, "CORRECTION"] = correction(error)

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(df["TOKENID"])
df.to_csv("test.csv")
preps = df[df["POS"] == "PRP"]
# words = df["TOKEN"].to_list()
X = []
corrected = preps[preps["TOKEN"] != preps["CORRECTION"]]
y = preps["CORRECTION"].to_list()

for i in range(preps.shape[0]):
    prep = preps.iloc[i]
    # d = {"prev": get_prev_word(prep, df),
    #      "curr": prep["TOKEN"],
    #      "next": get_next_word(prep, df)}

    dp_word = preps["TOKEN"][
        (preps["NID"] == prep["NID"]) & (preps["PID"] == prep["PID"]) & (preps["SID"] == prep["SID"]) & preps[
            "TOKENID"] == prep["DPHEAD"]]
    dp = {"dp_rel": prep["DPREL"], "dp_word": dp_word, "token": prep["TOKEN"]}
    # svm_dp = [prep["DPREL"], dp_word, prep["TOKEN"]]
    X.append(dp.copy())
    # svm_X.append(svm_dp.copy)

print(corrected)

vec = DictVectorizer(sparse=False)
X = vec.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = MultinomialNB().fit(X_train, y_train)
#clf = svm.SVC()
clf.fit(X_train, y_train)
# filtered = filtered[filtered["POS"] == "PRP"]
# filtered.to_csv("test.txt")

with open("test.txt", "w") as f:
    print(clf.score(X_test, y_test))
    scores = cross_val_score(clf, X, y, cv=5)
    prediction = clf.predict(X_test)
    changed = 0
    correct = 0
    inv = vec.inverse_transform(X_test)
    inv = [{k.split('=')[0]: k.split('=')[1] for k in row.keys()} for row in inv]

    for i in range(len(y_test)):
        tok = inv[i].get("token")
        # print(tok)
        w = str(tok) + "->" + str(prediction[i]) + " :: " + str(y_test[i]) + "\n"
        f.write(w)

        if str(tok) != str(y_test[i]):
            print(str(tok) + "->" + str(y_test[i]))
            changed += 1
            if prediction[i] == y_test[i]:
                correct += 1

print(str(correct) + "/" + str(changed))

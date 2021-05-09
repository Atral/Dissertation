from sklearn.model_selection import train_test_split

from load_data import *
from preprocess import *
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

# Loading data
SGML_PATH = "DATA/release3_2_PARSED/data/conll14st-preprocessed.conll.ann"
DATA_DIR = "DATA/release3_2_PARSED/data/conll14st-preprocessed.conll"
cols = ["NID", "PID", "SID", "TOKENID", "TOKEN", "POS", "DPHEAD", "DPREL", "SYNT", "CORRECTION"]
df = pd.read_table(DATA_DIR, encoding="ISO-8859-1", names=cols)
df["TOKEN"] = df["TOKEN"].str.lower()
# df = df.sample(n=round(df.shape[0]/10))
# Adding a correction column where the correction is equal to the token by default
df["CORRECTION"] = df["TOKEN"]



mistakes = find_mistakes(SGML_PATH)
prep_errors = filter_by_mistake_type(mistakes, "Prep")


# Adding corrections to the words which contain mistakes
print("Adding corrections...")
for error in prep_errors:
    # conditions = [df.NID == eval(error["nid"]), df.PID == eval(error["pid"]), df.SID == eval(error["sid"]),
    #               df.TOKENID >= eval(error["start_token"]), df.TOKENID < eval(error["end_token"])]
    mask = (df.NID == eval(error["nid"])) & (df.PID == eval(error["pid"])) & (df.SID == eval(error["sid"])) & \
           (df.TOKENID == eval(error["start_token"]))
    df.loc[mask, "CORRECTION"] = correction(error)
    # print(df.loc[mask])

df.to_csv("test.csv")
preps = df[df["DPREL"] == "prep"]
preps["CORRECTION"] = preps["CORRECTION"].str.lower()
# print(preps)
# print((preps['CORRECTION'].value_counts(ascending=False))[:20])

train, test = train_test_split(preps, test_size=0.1, random_state=42)

# # Filtering Uncommon Words
# top = ["of", "in", "to", "for", "on", "from", "with", "by", "as", "at", "into", "about", "like", "than", "through", "during", "according", "after", "over"]
# train = train[train["CORRECTION"].isin(top)]

# Undersampling the unchanged prepositions
print("Undersampling...")
changed = train[train["TOKEN"] != train["CORRECTION"]]
unchanged = train[train["TOKEN"] == train["CORRECTION"]]
# print(unchanged)
unchanged = unchanged.sample(n=round(unchanged.shape[0]*0.1), random_state=1)
frames = [changed, unchanged]
train = pd.concat(frames)
train = train.sample(frac=1)
X_train = []
y_train = train["CORRECTION"].to_list()


X_test = []
y_test = test["CORRECTION"].to_list()


def build_features(split, out):
    print("Building feature set...")
    for i in range(split.shape[0]):
        prep = split.iloc[i]
        # d = {"prev": get_prev_word(prep, df),
        #      "curr": prep["TOKEN"],
        #      "next": get_next_word(prep, df)}

        # mask = (split["NID"] == prep["NID"]) & (split["PID"] == prep["PID"]) & (split["SID"] == prep["SID"]) & split[
        #         "TOKENID"] == prep["DPHEAD"]
        mask2 = ((df["NID"] == prep["NID"]) & (df["PID"] == prep["PID"]) & (df["SID"] == prep["SID"]))
        dp_word = df.loc[mask2]
        head = dp_word[dp_word["TOKENID"] == eval(prep["DPHEAD"])]
        parent = head["TOKEN"].item()
        dp = {"token_id": prep["TOKENID"], "parent": parent, "token": prep["TOKEN"]}
        out.append(dp.copy())


build_features(train, X_train)
build_features(test, X_test)
X_train = (X_train[:len(X_test)])

print(len(X_train), len(X_test))

with open("base_train-shuffled", "a") as f:
    for index, item in enumerate(X_train):
        f.write(str(X_train[index]) + "|" + y_train[index] + "\n")

with open("base_test-shuffled", "a") as f:
    for index, item in enumerate(X_test):
        f.write(str(X_test[index]) + "|" + y_test[index] + "\n")

print("done")
import collections

from load_data import *
import pandas as pd
from timeit import default_timer as timer
from sklearn.feature_extraction import DictVectorizer

# Loading data
SGML_PATH = "DATA/release3_2_PARSED/data/conll14st-preprocessed.conll.ann"
DATA_DIR = "DATA/release3_2_PARSED/data/conll14st-preprocessed.conll"
cols = ["NID", "PID", "SID", "TOKENID", "TOKEN", "POS", "DPHEAD", "DPREL", "SYNT", "CORRECTION"]
df = pd.read_table(DATA_DIR, encoding="ISO-8859-1", names=cols)
# df = df.sample(n=round(df.shape[0]/10))
# Adding a correction column where the correction is equal to the token by default
df["CORRECTION"] = df["TOKEN"]
mistakes = find_mistakes(SGML_PATH)

mask2 = ((df.TOKEN == "of") | (df.TOKEN == "in") | (df.TOKEN == "for") | (df.TOKEN == "to") | (df.TOKEN == "on") |
         (df.TOKEN == "from") | (df.TOKEN == "with") | (df.TOKEN == "by") | (df.TOKEN == "as") | (df.TOKEN == "In") |
         (df.TOKEN == "at") | (df.TOKEN == "into") | (df.TOKEN == "about"))

pre_prep = df.loc[mask2]
print("filtering done")
print(pre_prep.shape[0])
filter_top = ["of", "in", "for", "to", "on", "", "from", "with", "by", "as", "In", "at", "into", "about"]
df2 = df.filter(mask2)
prep_errors = filter_by_mistake_type(df2, "Prep")
print(prep_errors.shape[0])

# Adding corrections to the words which contain mistakes
print("Adding corrections...")
for error in prep_errors:
    mask = (df.NID == eval(error["nid"])) & (df.PID == eval(error["pid"])) & (df.SID == eval(error["sid"])) & \
           (df.TOKENID == eval(error["start_token"]))
    df.loc[mask, "CORRECTION"] = correction(error)
    # print(df.loc[mask])

df.to_csv("test.csv")
preps = df2[df2["DPREL"] == "prep"]
# print(preps)
X = []

# Undersampling the unchanged prepositions
print("Undersampling...")
changed = preps[preps["TOKEN"] != preps["CORRECTION"]]
unchanged = preps[preps["TOKEN"] == preps["CORRECTION"]]
# print(unchanged)
unchanged = unchanged.sample(n=round(unchanged.shape[0] / 10))
frames = [changed, unchanged]
preps = pd.concat(frames)

y = preps["CORRECTION"].to_list()



print("Building feature set...")
for i in range(preps.shape[0]):
    prep = preps.iloc[i]
    # d = {"prev": get_prev_word(prep, df),
    #      "curr": prep["TOKEN"],
    #      "next": get_next_word(prep, df)}

    # mask = (preps["NID"] == prep["NID"]) & (preps["PID"] == prep["PID"]) & (preps["SID"] == prep["SID"]) & preps[
    #         "TOKENID"] == prep["DPHEAD"]
    mask2 = ((df["NID"] == prep["NID"]) & (df["PID"] == prep["PID"]) & (df["SID"] == prep["SID"]))
    dp_word = df.loc[mask2]
    head = dp_word[dp_word["TOKENID"] == eval(prep["DPHEAD"])]
    parent = head["TOKEN"].item()
    dp = {"token_id": prep["TOKENID"], "parent": parent, "token": prep["TOKEN"]}
    X.append(dp.copy())

with open("processed_remov.txt", "w") as f:
    for index, item in enumerate(X):
        f.write(str(X[index]) + "|" + y[index] + "\n")
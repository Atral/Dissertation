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
prep_errors = filter_by_mistake_type(mistakes, "Prep")

mistakes = find_mistakes(SGML_PATH)
mask2 = ((prep_errors["TOKEN"] == "of") | (prep_errors["TOKEN"] == "in") | (prep_errors["TOKEN"] == "for") | (prep_errors["TOKEN"] == "to")
                          | (prep_errors["TOKEN"] == "on") | (prep_errors["TOKEN"] == "from") | (prep_errors["TOKEN"] == "with") |
                          prep_errors["TOKEN"] == "by" | prep_errors["TOKEN"] == "as" | prep_errors["TOKEN"] == "In"
                          |( prep_errors["TOKEN"] == "at") | (prep_errors["TOKEN"] == "into") | (prep_errors["TOKEN"] == "about"))
pre_prep = df.loc[mask2]
print("filtering done")
print(pre_prep)
filter_top = ["of", "in", "for", "to", "on", "","from", "with", "by", "as", "In", "at", "into", "about"]


prep_errors = df.filter(mask2)
print(prep_errors)

print(prep_errors)

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
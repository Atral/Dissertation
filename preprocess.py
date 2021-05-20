from sklearn.model_selection import train_test_split
from load_data import *
import pandas as pd
from sklearn.feature_extraction import DictVectorizer


# Adding corrections to the words which contain mistakes
def add_corrections(errors):
    print("Adding corrections...")
    for error in errors:
        mask = (df.NID == eval(error["nid"])) & (df.PID == eval(error["pid"])) & (df.SID == eval(error["sid"])) & \
               (df.TOKENID == eval(error["start_token"]))
        df.loc[mask, "CORRECTION"] = correction(error).lower()
        # print(df.loc[mask])


def filter_uncommon(list, frame):
    # Filtering Uncommon Words
    return frame[frame["CORRECTION"].isin(list)]


def remove_word(word, frame):
    removed = frame[frame["CORRECTION"]!= word]
    return removed


def undersample(frame, amount):
    print("Undersampling...")
    # Separating changed and unchanged data
    changed = frame[frame["TOKEN"] != frame["CORRECTION"]]
    unchanged = frame[frame["TOKEN"] == frame["CORRECTION"]]
    orig = unchanged.shape[0]
    print("changed to unchanged ratio = " + str(orig/(frame.shape[0])))
    # Undersampling
    unchanged = unchanged.sample(n=round(amount*changed.shape[0]), random_state=1).reset_index(drop=True)
    # Rejoining
    frames = [changed, unchanged]
    frame = pd.concat(frames)

    new = unchanged.shape[0]
    print("Downsampled unchanged data from " + str(orig) + " to " + str(new) + " at " + str(new / orig))
    return frame.sample(frac=1).reset_index(drop=True)


def build_features(split):
    print("Building feature set...")
    out = []
    for i in range(split.shape[0]):
        prep = split.iloc[i]
        mask2 = ((df["NID"] == prep["NID"]) & (df["PID"] == prep["PID"]) & (df["SID"] == prep["SID"]))
        dp_word = df.loc[mask2]
        head = dp_word[dp_word["TOKENID"] == eval(prep["DPHEAD"])]
        parent = head["TOKEN"].item()
        parent_pos = head["POS"].item()

        dp = {"tokenID": prep["TOKENID"], "parent": parent, "parent_pos": parent_pos, "token": prep["TOKEN"],
              "tok-1": get_prev_word(prep, df),"tok+1": get_next_word(prep, df),
              "pos-1": get_n_pos(prep, df, -1), "pos+1": get_n_pos(prep, df, 1)}
        out.append(dp.copy())
    return out


def balance_test_train(tst, tr):
    tst_size = test.shape[0]
    tr_size = train.shape[0]

    if tst_size > tr_size:
        tst.sample(n=tr_size)
    else:
        tr.sample(n=tst_size)
    return [tst, tr]


def write_to_file(filename, x_group, y_group):
    with open(filename, "w") as f:
        for index, item in enumerate(x_group):
            f.write(str(x_group[index]) + "|" + y_group[index] + "\n")


# Loading data
SGML_PATH = "DATA/release3_2_PARSED/data/conll14st-preprocessed.conll.ann"
DATA_DIR = "DATA/release3_2_PARSED/data/conll14st-preprocessed.conll"
cols = ["NID", "PID", "SID", "TOKENID", "TOKEN", "POS", "DPHEAD", "DPREL", "SYNT", "CORRECTION"]
df = pd.read_table(DATA_DIR, encoding="ISO-8859-1", names=cols)
df["TOKEN"] = df["TOKEN"].str.lower()
# Adding a correction column where the correction is equal to the token by default
df.dropna(subset=["TOKEN"], inplace=True)
df["CORRECTION"] = df["TOKEN"]
df.dropna(subset=["CORRECTION"], inplace=True)

# Getting preposition error set
mistakes = find_mistakes(SGML_PATH)
prep_errors = filter_by_mistake_type(mistakes, "Prep")
add_corrections(prep_errors)

# Finding prepositions in text
preps = df[df["DPREL"] == "prep"]
counts = preps["CORRECTION"].value_counts().index.to_list()
t = preps["CORRECTION"].value_counts()
print(t.head(30))
undersample_rate = 2

# Split into test and train data
train, test = train_test_split(preps, test_size=0.15, random_state=42)

# Undersample the train data

top = counts[:20]
filter_uncommon(top, train)

train = undersample(train, undersample_rate)
p = train[train["CORRECTION"] == train["TOKEN"]].shape[0]
f = train[train["CORRECTION"] != train["TOKEN"]].shape[0]
print(f/p)


print(test.shape[0], train.shape[0])

# # Balance test and train data
# balanced = balance_test_train(test, train)
# test = balanced[0]
# train = balanced[1]

# Build test and train features
X_train = build_features(train)
y_train = train["CORRECTION"].to_list()

X_test = build_features(test)
y_test = test["CORRECTION"].to_list()

# Write data to file
write_to_file("top20_test.txt", X_test, y_test)
write_to_file("top20_train.txt", X_train, y_train)

print("done")

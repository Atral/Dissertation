from load_data import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

SGML_PATH = "DATA/release3_2_PARSED/data/conll14st-preprocessed.conll.ann"
DATA_DIR = "DATA/release3_2_PARSED/data/conll14st-preprocessed.conll"
cols = ["NID", "PID", "SID", "TOKENID", "TOKEN", "POS", "DPHEAD", "DPREL", "SYNT", "CORRECTION"]

df = pd.read_table(DATA_DIR, encoding="ISO-8859-1", names=cols)
df["CORRECTION"] = "*correct*"

#paragraphs = find_paragraphs(SGML_PATH)
mistakes = find_mistakes(SGML_PATH)
prep_errors = filter_by_mistake_type(mistakes, "Prep")

# labels = []
# adj_words = get_adjacent_words(prep_errors[2], df)
# incorrect_string = get_error_string(prep_errors[2], df)
# error_loc = get_error_position(prep_errors[2], df)
# sentence = get_sentence(prep_errors[2], df)
# correction = correction(prep_errors[2])
# preps["CORRECTION"] = 1


for error in prep_errors:
    mask = (df.NID == eval(error["nid"]))
    mask &= (df.PID == eval(error["pid"]))
    mask &= (df.SID == eval(error["sid"]))
    mask &= (df.TOKENID >= eval(error["start_token"]))
    mask &= (df.TOKENID <= eval(error["start_token"]))
    df.loc[mask, "CORRECTION"] = correction(error)

with open("text.txt", "w", encoding="utf-8") as f:
    #preps = df[(df["DPREL"] == "prep")]
    words = df["TOKEN"].to_list()
    x = []
    labels = df["CORRECTION"].to_list()

    i = 0
    while i < len(labels)-1:
        adj = str(words[i-1]) + " " + str(words[i]) + " " + str(words[i+1])
        x.append(adj)
        #print(str(i) + "/" + str(len(labels)))
        if labels[i] != "*correct*":
            f.write(str(words[i]) + ">" + str(labels[i]) + " ")
        else:
            f.write(str(words[i]) + " ")
        i += 1

    labels = labels[:-1]
    print(x)

    X_train, X_test, y_train, y_test = train_test_split(x, labels, test_size=0.2)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    clf = MultinomialNB().fit(X_train_tfidf, y_train)
    print(clf.predict("need of habitable"))








    # paragraph = document[(document["PID"] == eval(error["pid"]))]
    # sentence = paragraph[(paragraph["SID"] == eval(error["sid"]))]
    # errors = sentence[(sentence["TOKENID"] >= eval(error["start_token"])) & (
    #         sentence["TOKENID"] <= eval(error["end_token"]))]
    # if not errors.empty:
    #     df.loc[(df["DPREL"] == "prep" & df["NID"] == eval(error["nid"])), "CORRECTION"] = correction(error)


#sentence[error_loc[0]:error_loc[len(error_loc
#print("correcting mistakes")

#with open("test.txt", "w") as f:
    #for error in prep_errors:

        #labels.append(correction(error))
       # print(get_adjacent_words(df, error))

# x = find_corrections(, prep_errors)
# print(x)


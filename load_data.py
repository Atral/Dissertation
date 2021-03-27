import pandas as pd
from bs4 import BeautifulSoup

DATA_DIR = "DATA/release3_2_PARSED/data/conll14st-preprocessed.conll"
cols = ["NID", "PID", "SID", "TOKENID", "TOKEN", "POS", "DPHEAD", "DPREL", "SYNT"]
df = pd.read_table(DATA_DIR, encoding="ISO-8859-1", names=cols)
SGML_DIR = "DATA/release3_2_PARSED/data/conll14st-preprocessed.conll.ann"


def find_paragraphs(sgml_path):
    print("finding paragraphs...")
    with open(sgml_path, "r") as file:
        soup = BeautifulSoup(file, "html.parser")
        paragraphs = soup.find_all("p")
    return paragraphs


def find_mistakes(sgml_path):
    print("finding mistakes...")
    with open(sgml_path, "r") as file:
        soup = BeautifulSoup(file, "html.parser")
        mistakes = soup.find_all("mistake")
    return mistakes


def start_par(mistake):
    return eval(mistake["start_par"]) - 1


def end_par(mistake):
    return eval(mistake["end_par"]) - 1


def get_start_token(mistake):
    return eval(mistake["start_token"])


def get_end_token(mistake):
    return eval(mistake["end_token"])


def get_errors(mistake, filtered_table):
    document = filtered_table[(filtered_table["NID"] == eval(mistake["nid"]))]
    paragraph = document[(document["PID"] == eval(mistake["pid"]))]
    sentence = paragraph[(paragraph["SID"] == eval(mistake["sid"]))]
    errors = sentence[(sentence["TOKENID"] >= eval(mistake["start_token"])) & (sentence["TOKENID"] <= eval(mistake["end_token"]))]
    return errors


def get_error_string(mistake, filtered_table):
    error = get_errors(mistake, filtered_table)
    return error["TOKEN"].tolist()


def get_error_position(mistake, filtered_table):
    error = get_errors(mistake, filtered_table)
    return error["TOKENID"].tolist()


def get_sentence(mistake, filtered_table):
    document = filtered_table[(filtered_table["NID"] == eval(mistake["nid"]))]
    paragraph = document[(document["PID"] == eval(mistake["pid"]))]
    sentence = paragraph[(paragraph["SID"] == eval(mistake["sid"]))]
    return sentence["TOKEN"].tolist()


def start_mistake(mistake):
    return eval(mistake["start_off"])


def end_mistake(mistake):
    return eval(mistake["end_off"]) + 1


def err_type(mistake):
    return mistake.find_next("type")


def correction(mistake):
    return mistake.find_next("correction").text


def get_start_token(mistake):
    return mistake["start_token"]


def get_end_token(mistake):
    return mistake["end_token"]


def inc_str(paragraph, mistake):
    return paragraph[start_par(mistake)].text[start_mistake(mistake):end_mistake(mistake)]


def inc_paragraph(paragraph, mistake):
    return paragraph[start_par(mistake)].text


def corr_paragraph(paragraph, mistake):
    inc_para = inc_paragraph(paragraph, mistake)
    corrected = inc_para[:start_mistake(mistake) + 1] + correction(
        mistake) + inc_para[end_mistake(mistake):]
    return corrected


def filter_by_mistake_type(mistakes, error_type):
    filtered_mistakes = []
    for mistake in mistakes:
        if mistake.findChild("type", text=error_type):
            filtered_mistakes.append(mistake)
    return filtered_mistakes


def filter_except_type(mistakes, error_type):
    filtered_mistakes = []
    for mistake in mistakes:
        if mistake.findChild("type", text=error_type):
            filtered_mistakes = filtered_mistakes
        else:
            filtered_mistakes.append(mistakes)
    return filtered_mistakes


def get_adjacent_words(mistake, data_field):
    document = data_field[(data_field["NID"] == eval(mistake["nid"]))]
    paragraph = document[(document["PID"] == eval(mistake["pid"]))]
    sentence = paragraph[(paragraph["SID"] == eval(mistake["sid"]))]
    prev_word = sentence[(sentence["TOKENID"]) == eval(mistake["start_token"]) - 1]
    next_word = sentence[(sentence["TOKENID"]) == eval(mistake["end_token"]) + 1]
    error_words = get_error_string(mistake, data_field)
    all_words = [prev_word["TOKEN"].to_string(index=False)] + error_words + [next_word["TOKEN"].to_string(index=False)]
    return all_words


def get_table_token(data_field):
    return data_field["TOKEN"]


def filter_table_by_pos(data_field, pos_tag):
    filtered = data_field[(data_field["POS"] == pos_tag)]
    return filtered


def filter_table_by_dprel(data_field, dprel):
    return data_field[(data_field["DPREL"] == dprel)]


def get_parent_pos(data_field):
    parent_pos = data_field["DPHEAD"]
    return parent_pos


def get_dependency_relationship(data_field):
    dependency_relationship = data_field["DPREL"]
    return dependency_relationship

# print(table_by_pos(df, "PRP"))
# print(df["NID"])
# row = df.loc[3]
# print(row["TOKEN"])
# mistakes = find_mistakes(SGML_DIR)
# x = get_adjacent_words(mistakes[0], df)
# print(mistakes[0])
# print(get_adjacent_words(mistakes[0], df))

import pandas as pd
from bs4 import BeautifulSoup

DATA_DIR = "DATA/release3_2_PARSED/data/conll14st-preprocessed.conll"
cols = ["NID", "PID", "SID", "TOKENID", "TOKEN", "POS", "DPHEAD", "DPREL", "SYNT"]
df = pd.read_table(DATA_DIR, encoding="ISO-8859-1", names=cols)
SGML_DIR = "DATA/release3_2_PARSED/data/nucle3.2.sgml"


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


def start_mistake(mistake):
    return eval(mistake["start_off"])


def end_mistake(mistake):
    return eval(mistake["end_off"]) + 1


def err_type(mistake):
    return mistake.find_next("type")


def correction(mistake):
    return mistake.find_next("correction").text


def inc_str(paragraph, mistake):
    return paragraph[start_par(mistake)].text[start_mistake(mistake):end_mistake(mistake)]


def inc_paragraph(paragraph, mistake):
    return paragraph[start_par(mistake)].text


def corr_paragraph(paragraph, mistake):
    inc_para = inc_paragraph(paragraph, mistake)
    corrected = inc_para[:start_mistake(mistake)+1] + correction(
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

def get_adjacent_words():
    print("")
    # i = 0
    # for mistake in mistakes:
    #     print(str(i) + "/" + str(len(mistakes)) + " filtering error")
    #     i += 1
    #     if err_type(mistake) == error_type:
    #         print("matching type found")
    #         filtered.append(mistake)

    # return filtered
print(df["NID"])

# with open("DATA/release3_2_PARSED/data/nucle3.2.sgml", "r") as file:
#     soup = BeautifulSoup(file, "html.parser")
#     types = soup.find_all("type", text="Prep")
#     #filtered_list = types.findPreviousSiblings
#     print(types)


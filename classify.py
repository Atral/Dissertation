from load_data import *
SGML_PATH = "DATA/release3_2_PARSED/data/nucle3.2.sgml"

paragraphs = find_paragraphs(SGML_PATH)
mistakes = find_mistakes(SGML_PATH)
prep_errors = filter_by_mistake_type(mistakes, "Prep")
other_errors = filter_except_type(mistakes, "Prep")


error_strings = []
corrected_strings = []


print("correcting mistakes")

with open("test.txt", "w") as f:
    for error in prep_errors:
        corrected = corr_paragraph(paragraphs, error)
        corrected_strings.append(corrected)
        f.write(corrected)


print(len(paragraphs))
print(len(corrected_strings))




import sys
import nltk
import numpy as np
from nltk.corpus.reader.conll import ConllCorpusReader

def get_input(filepath):
    f = open(filepath, 'r')
    content = f.read()
    return content

PATH = "DATA/release3_2_PARSED/data"
FILEID = "conll14st-preprocessed.conll"

#content = get_input(PATH)
#print(content)

data = ConllCorpusReader(PATH, fileids=FILEID, columntypes=['chunk','chunk','chunk','chunk','words','pos','chunk','chunk','tree'],
                         root_label='S', pos_in_tree=True, encoding='utf8', separator=' ')

#print(data.chunked_words(fileids=None, chunk_types=None, tagset=None))
print(data.raw())
#print(data.words())

from nltk.corpus.reader.conll import ConllCorpusReader

crtest = ConllCorpusReader('E:\\Downloads\\Documents\\Uni\\Dissertation\\DATA\\release2_3_1\\original\\data\\',
                           ['official-preprocessed.conll'],
                            ['words','pos','chunk'],
                            encoding='utf8')

pprint(verb)
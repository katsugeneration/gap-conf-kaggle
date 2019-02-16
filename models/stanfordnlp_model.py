import pandas
import utils
from collections import namedtuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer


dtype = np.int32
START_OF_SENTENCE = "SOS"
END_OF_SENTENCE = "EOS"
POS_TYPES = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB"]
cv = CountVectorizer(dtype=dtype)
cv.fit(POS_TYPES + [START_OF_SENTENCE, END_OF_SENTENCE])


def _get_bag_of_pos(words, index, N):
    """Return pos list surrounding index
    Args:
        words (list): stanfordnlp word list object having pos attributes.
        index (int): target index
        N (int): return +/- N word pos
    Return:
        pos_list (List[str]): xpo format string list
    """
    DummyWord = namedtuple("DummyWord", "pos")
    sos = DummyWord(pos=START_OF_SENTENCE)
    eos = DummyWord(pos=END_OF_SENTENCE)
    words = [sos] * N + words + [eos] * N
    index += N
    return [w.pos for w in words[index-N:index+N+1]]


def _vectorise_bag_of_pos(words, indexes, N):
    """Return pos list surrounding index
    Args:
        words (list): stanfordnlp word list object having pos attributes.
        indexes (List[int]): target indexes
        N (int): return +/- N word pos
    Return:
        pos_list (List[str]): xpo format string list
    """
    matrixes = []
    for index in indexes:
        poss = _get_bag_of_pos(words, index, N)
        matrixes.append(" ".join(poss))
    return cv.transform(matrixes).toarray().flatten()


def _get_classify_labels(df):
    """Return task classify label

    if A-coref is True, return 0
    if B-coref is True, return 1
    if A-coref and B-coref is Fale, return 2
    Args:
        df (DataFram): pandas DataFrame object
    Return:
        labels (array): label values. type of numpy int32 array. shaep is (N, 1)
    """
    labels = np.ones((len(df), 1), dtype=dtype) * 2
    labels[df['A-coref']] = 0
    labels[df['B-coref']] = 1
    return labels


def evaluate(test_data):
    df = pandas.read_csv('dataset/gap-development.tsv', sep='\t')

    X = []
    for i in range(10):
        words, pronnoun_index = utils.charpos_to_word_index(df['Text'][i], df['Pronoun-offset'][i])
        words, A_index = utils.charpos_to_word_index(df['Text'][i], df['A-offset'][i])
        words, B_index = utils.charpos_to_word_index(df['Text'][i], df['B-offset'][i])
        X.append(_vectorise_bag_of_pos(words, [pronnoun_index, A_index, B_index], 5))
    print(X)
    print(PCA(n_components=2).fit_transform(X))


evaluate('')

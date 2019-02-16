import pandas
import utils
import pickle
from collections import namedtuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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


def _preprocess_data(df):
    """Preprocess task speccific pipeline.
    Args:
        df (DataFrame): target pandas DataFrame object.
    Return:
        X (array): explanatory variables in task. shape is (n_sumples, n_features)
        Y (array): objective variables in task. shape is (n_sumples, 1)
    """
    X = []
    Y = _get_classify_labels(df)
    for i in range(len(df)):
        words, pronnoun_index = utils.charpos_to_word_index(df['Text'][i], df['Pronoun-offset'][i])
        _, A_index = utils.charpos_to_word_index(df['Text'][i], df['A-offset'][i], words=words)
        _, B_index = utils.charpos_to_word_index(df['Text'][i], df['B-offset'][i], words=words)
        X.append(_vectorise_bag_of_pos(words, [pronnoun_index, A_index, B_index], 5))
    return X, Y


def train(use_preprocessdata=False):
    if use_preprocessdata:
        try:
            with open('preprocess_traindata.pkl', 'rb') as f:
                X, Y = pickle.load(f)
        except:  # noqa
            use_preprocessdata = False

    if not use_preprocessdata:
        df = pandas.read_csv('dataset/gap-test.tsv', sep='\t')
        X, Y = _preprocess_data(df)
        print("Data Loaded")

        with open('preprocess_traindata.pkl', 'wb') as f:
            pickle.dump((X, Y), f, protocol=pickle.HIGHEST_PROTOCOL)
    lr = LogisticRegression(random_state=0)
    lr.fit(X, Y)
    with open('model.pkl', 'wb') as f:
        pickle.dump(lr, f, protocol=pickle.HIGHEST_PROTOCOL)
    y_pred = lr.predict(X)
    print("Train Accuracy:", accuracy_score(Y, y_pred))


def evaluate(test_data, use_preprocess_testdata=True):
    if use_preprocess_testdata:
        try:
            with open('preprocess_testdata.pkl', 'rb') as f:
                X, Y = pickle.load(f)
        except:  # noqa
            use_preprocess_testdata = False

    if not use_preprocess_testdata:
        X, Y = _preprocess_data(test_data)
        print("Data Loaded")

        with open('preprocess_testdata.pkl', 'wb') as f:
            pickle.dump((X, Y), f, protocol=pickle.HIGHEST_PROTOCOL)

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    y_pred = model.predict(X)
    print("Test Accuracy:", accuracy_score(Y, y_pred))

    predicts = model.predict_proba(X)
    out_df = pandas.DataFrame(data=predicts, columns=['A', 'B', 'NEITHER'])
    out_df['ID'] = test_data['ID']
    return out_df


train(use_preprocessdata=True)

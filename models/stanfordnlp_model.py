import pandas
import utils
import pickle
from collections import namedtuple
import itertools
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer


dtype = np.int32
START_OF_SENTENCE = "SOS"
END_OF_SENTENCE = "EOS"
DEFAULT_NGRAM_WINDOW = 2
POS_TYPES = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WRB"]
cv_normal = CountVectorizer(dtype=dtype)
cv_normal.fit(POS_TYPES + [START_OF_SENTENCE, END_OF_SENTENCE])
cv_ngram = CountVectorizer(dtype=dtype)
cv_ngram.fit(["_".join(p) for p in itertools.product(
    POS_TYPES + [START_OF_SENTENCE, END_OF_SENTENCE],
    repeat=DEFAULT_NGRAM_WINDOW)])
print(len(cv_ngram.vocabulary_))


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
    return cv_normal.transform(matrixes).toarray().flatten()


def _get_bag_of_pos_ngram(words, index, window_size, N):
    """Return pos list surrounding index
    Args:
        words (list): stanfordnlp word list object having pos attributes.
        index (int): target index
        window_size (int): target window size return +/- N word pos n-grma
        N (int): n-gram set
    Return:
        pos_list (List[str]): xpo format string list
    """
    DummyWord = namedtuple("DummyWord", "pos")
    sos = DummyWord(pos=START_OF_SENTENCE)
    eos = DummyWord(pos=END_OF_SENTENCE)
    words = [sos] * (window_size + N) + words + [eos] * (window_size + N)
    index += (window_size + N)
    return [
        "_".join([w.pos.replace('$', '') for w in words[i:i+N]])
        for i in range(index-window_size, index+window_size+1)]


def _vectorise_bag_of_pos_ngram(words, indexes, window_size, N=DEFAULT_NGRAM_WINDOW):
    """Return pos list surrounding index
    Args:
        words (list): stanfordnlp word list object having pos attributes.
        indexes (List[int]): target indexes
        window_size (int): target window size return +/- N word pos n-grma
        N (int): n-gram set
    Return:
        pos_list (List[str]): xpo format string list
    """
    matrixes = []
    for index in indexes:
        poss = _get_bag_of_pos_ngram(words, index, window_size, N)
        matrixes.append(" ".join(poss))
    return cv_ngram.transform(matrixes).toarray().flatten()


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


def _preprocess_data(df, use_preprocessdata=False, save_path=None):
    """Preprocess task speccific pipeline.
    Args:
        df (DataFrame): target pandas DataFrame object.
        use_preprocessdata (bool): Wheter or not to use local preprocess file loading
        save_path (str): local preprocess file path
    Return:
        X (array): explanatory variables in task. shape is (n_sumples, n_features)
        Y (array): objective variables in task. shape is (n_sumples, 1)
    """
    if use_preprocessdata:
        try:
            with open(save_path, 'rb') as f:
                data = pickle.load(f)
        except:  # noqa
            use_preprocessdata = False

    if not use_preprocessdata:
        data = []
        for i in range(len(df)):
            words, pronnoun_index = utils.charpos_to_word_index(df['Text'][i], df['Pronoun-offset'][i])
            _, A_index = utils.charpos_to_word_index(df['Text'][i], df['A-offset'][i], words=words)
            _, B_index = utils.charpos_to_word_index(df['Text'][i], df['B-offset'][i], words=words)
            data.append((words, [pronnoun_index, A_index, B_index]))
        with open(save_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Data Loaded")

    X = []
    for (words, indexes) in data:
        X.append(_vectorise_bag_of_pos(words, indexes, 3))
    Y = _get_classify_labels(df)
    return X, Y


def train(use_preprocessdata=True):
    df = pandas.read_csv('dataset/gap-test.tsv', sep='\t')
    X, Y = _preprocess_data(df, use_preprocessdata=use_preprocessdata, save_path='preprocess_traindata.pkl')
    model = LogisticRegression(random_state=0)
    # model = SVC(C=10, probability=True, random_state=0)
    # model = MLPClassifier(hidden_layer_sizes=(50, 30, 30, 50), activation='relu', solver='adam', batch_size=128, random_state=0)
    model.fit(X, Y)
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    y_pred = model.predict(X)
    print("Train Accuracy:", accuracy_score(Y, y_pred))


def evaluate(test_data, use_preprocessdata=True):
    X, Y = _preprocess_data(test_data, use_preprocessdata=use_preprocessdata, save_path='preprocess_testdata.pkl')
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    y_pred = model.predict(X)
    print("Test Accuracy:", accuracy_score(Y, y_pred))

    predicts = model.predict_proba(X)
    out_df = pandas.DataFrame(data=predicts, columns=['A', 'B', 'NEITHER'])
    out_df['ID'] = test_data['ID']
    return out_df


# train()

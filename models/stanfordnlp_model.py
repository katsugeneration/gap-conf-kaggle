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
import xgboost as xgb


dtype = np.int32
DEFAULT_NGRAM_WINDOW = 2
DEFAULT_WINDOW_SIZE = 5

DummyWord = namedtuple("DummyWord", "pos")
cv_normal = CountVectorizer(dtype=dtype)
cv_normal.fit(utils.POS_TYPES + [utils.START_OF_SENTENCE, utils.END_OF_SENTENCE])
cv_ngram = CountVectorizer(dtype=dtype)
cv_ngram.fit(["_".join(p) for p in itertools.product(
    utils.POS_TYPES + [utils.START_OF_SENTENCE, utils.END_OF_SENTENCE],
    repeat=DEFAULT_NGRAM_WINDOW)])
cv_position = CountVectorizer(token_pattern=r'\b[-\w][-\w]+\b', dtype=dtype)
cv_position.fit([p[0] + "_" + str(p[1]) for p in itertools.product(
    utils.POS_TYPES + [utils.START_OF_SENTENCE, utils.END_OF_SENTENCE], range(-DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_SIZE+1))])


def _get_bag_of_pos(words, index, N):
    """Return pos list surrounding index
    Args:
        words (list): stanfordnlp word list object having pos attributes.
        index (int): target index
        N (int): return +/- N word pos
    Return:
        pos_list (List[str]): xpo format string list
    """
    sos = DummyWord(pos=utils.START_OF_SENTENCE)
    eos = DummyWord(pos=utils.END_OF_SENTENCE)
    words = [sos] * N + words + [eos] * N
    index += N
    return [w.pos.replace('$', '') for w in words[index-N:index+N+1]]


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
    sos = DummyWord(pos=utils.START_OF_SENTENCE)
    eos = DummyWord(pos=utils.END_OF_SENTENCE)
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


def _get_bag_of_pos_with_position(words, index, N):
    """Return pos list surrounding index
    Args:
        words (list): stanfordnlp word list object having pos attributes.
        index (int): target index
        N (int): return +/- N word pos
    Return:
        pos_list (List[str]): xpo format string list
    """
    sos = DummyWord(pos=utils.START_OF_SENTENCE)
    eos = DummyWord(pos=utils.END_OF_SENTENCE)
    words = [sos] * N + words + [eos] * N
    index += N
    return [w.pos.replace('$', '') + '_' + str(i-N) for i, w in enumerate(words[index-N:index+N+1])]


def _vectorise_bag_of_pos_with_position(words, indexes, N):
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
        poss = _get_bag_of_pos_with_position(words, index, N)
        matrixes.append(" ".join(poss))
    return cv_position.transform(matrixes).toarray().flatten()


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
        X.append(_vectorise_bag_of_pos_with_position(words, indexes, DEFAULT_WINDOW_SIZE))
    X = np.array(X)
    Y = _get_classify_labels(df)
    return X, Y


def train(use_preprocessdata=True):
    df = pandas.read_csv('dataset/gap-test.tsv', sep='\t')
    X, Y = _preprocess_data(df, use_preprocessdata=use_preprocessdata, save_path='preprocess_traindata.pkl')
    validation_df = pandas.read_csv('dataset/gap-validation.tsv', sep='\t')
    validation_X, validation_Y = _preprocess_data(validation_df, use_preprocessdata=use_preprocessdata, save_path='preprocess_valdata.pkl')
    # model = LogisticRegression(random_state=0)
    model = xgb.XGBClassifier(max_depth=5, n_jobs=8, random_state=0)
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


train()

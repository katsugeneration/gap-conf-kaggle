import pandas
import utils
import pickle
from collections import namedtuple
import itertools
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.feature_extraction.text import CountVectorizer
import optuna
import xgboost as xgb


dtype = np.int32
DEFAULT_NGRAM_WINDOW = 2
DEFAULT_WINDOW_SIZE = 10
NONE_DEPENDENCY = 'NONE'

DummyWord = namedtuple("DummyWord", "pos upos")
cv_normal = CountVectorizer(dtype=dtype)
cv_normal.fit(utils.POS_TYPES + [utils.BEGIN_OF_SENTENCE, utils.END_OF_SENTENCE])
cv_ngram = CountVectorizer(dtype=dtype)
cv_ngram.fit(["_".join(p) for p in itertools.product(
    utils.POS_TYPES + [utils.BEGIN_OF_SENTENCE, utils.END_OF_SENTENCE],
    repeat=DEFAULT_NGRAM_WINDOW)])
cv_position = CountVectorizer(token_pattern=r'\b[-\w][-\w]+\b', dtype=dtype)
cv_position.fit([p[0] + "_" + str(p[1]) for p in itertools.product(
    utils.POS_TYPES + [utils.BEGIN_OF_SENTENCE, utils.END_OF_SENTENCE], range(-DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_SIZE+1))])
cv_upos_position = CountVectorizer(token_pattern=r'\b[-\w][-\w]+\b', dtype=dtype)
cv_upos_position.fit([p[0] + "_" + str(p[1]) for p in itertools.product(
    utils.UPOS_TYPES + [utils.BEGIN_OF_SENTENCE, utils.END_OF_SENTENCE], range(-DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_SIZE+1))])
dependencies = ['governor', 'child', 'ancestor', 'grandchild']
cv_dependencies = CountVectorizer(token_pattern=r'\b[-\w][-\w]+\b', dtype=dtype)
cv_dependencies.fit([p[0] + "_" + str(p[1]) for p in itertools.product(
    utils.POS_TYPES + [NONE_DEPENDENCY], dependencies)])


def _get_bag_of_pos(words, index, N, target_len=1):
    """Return pos list surrounding index
    Args:
        words (list): stanfordnlp word list object having pos attributes.
        index (int): target index
        N (int): return +/- N word pos
        target_len (int): target word length
    Return:
        pos_list (List[str]): xpo format string list
    """
    bos = DummyWord(pos=utils.BEGIN_OF_SENTENCE, upos=utils.BEGIN_OF_SENTENCE)
    eos = DummyWord(pos=utils.END_OF_SENTENCE, upos=utils.END_OF_SENTENCE)
    words = [bos] * N + words + [eos] * N
    index += N
    return [w.pos.replace('$', '') for w in words[index-N:index] + [words[index]] + words[index+target_len:index+target_len+N]]


def _vectorise_bag_of_pos(words, indexes, N, targets=[]):
    """Return pos list surrounding index
    Args:
        words (list): stanfordnlp word list object having pos attributes.
        indexes (List[int]): target indexes
        N (int): return +/- N word pos
        target_len (int): target word length
    Return:
        pos_list (List[str]): xpo format string list
    """
    matrixes = []
    for i, index in enumerate(indexes):
        poss = _get_bag_of_pos(words, index, N, target_len=len(targets[i].split()))
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
    bos = DummyWord(pos=utils.BEGIN_OF_SENTENCE, upos=utils.BEGIN_OF_SENTENCE)
    eos = DummyWord(pos=utils.END_OF_SENTENCE, upos=utils.END_OF_SENTENCE)
    words = [bos] * (window_size + N) + words + [eos] * (window_size + N)
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


def _get_bag_of_pos_with_position(words, index, N, target_len=1):
    """Return pos list surrounding index
    Args:
        words (list): stanfordnlp word list object having pos attributes.
        index (int): target index
        N (int): return +/- N word pos
        target_len (int): target word length
    Return:
        pos_list (List[str]): xpo format string list
    """
    bos = DummyWord(pos=utils.BEGIN_OF_SENTENCE, upos=utils.BEGIN_OF_SENTENCE)
    eos = DummyWord(pos=utils.END_OF_SENTENCE, upos=utils.END_OF_SENTENCE)
    words = [bos] * N + words + [eos] * N
    index += N
    return [w.pos.replace('$', '') + '_' + str(i-N) for i, w in enumerate(
        words[index-N:index] + [words[index]] + words[index+target_len:index+target_len+N])]


def _vectorise_bag_of_pos_with_position(words, indexes, N, targets=[]):
    """Return pos list surrounding index
    Args:
        words (list): stanfordnlp word list object having pos attributes.
        indexes (List[int]): target indexes
        N (int): return +/- N word pos
        targets (List[str]): target word list. length is same as indexes
    Return:
        pos_list (List[str]): xpo format string list
    """
    matrixes = []
    for i, index in enumerate(indexes):
        poss = _get_bag_of_pos_with_position(words, index, N, target_len=len(targets[i].split()))
        matrixes.append(" ".join(poss))
    return cv_position.transform(matrixes).toarray().flatten()


def _get_bag_of_upos_with_position(words, index, N, target_len=1):
    """Return pos list surrounding index
    Args:
        words (list): stanfordnlp word list object having pos attributes.
        index (int): target index
        N (int): return +/- N word pos
        target_len (int): target word length
    Return:
        pos_list (List[str]): upos format string list
    """
    bos = DummyWord(pos=utils.BEGIN_OF_SENTENCE, upos=utils.BEGIN_OF_SENTENCE)
    eos = DummyWord(pos=utils.END_OF_SENTENCE, upos=utils.END_OF_SENTENCE)
    words = [bos] * N + words + [eos] * N
    index += N
    return [w.upos.replace('$', '') + '_' + str(i-N) for i, w in enumerate(
        words[index-N:index] + [words[index]] + words[index+target_len:index+target_len+N])]


def _get_bag_of_pos_with_dependency(words, index):
    """Return pos list surrounding index
    Args:
        words (list): stanfordnlp word list object having pos attributes.
        index (int): target index
    Return:
        pos_list (List[str]): xpos format string list
    """
    pos_list = []

    def _get_governor(_index, name):
        governor_list = []
        governor_index = _index + (int(words[_index].governor) - int(words[_index].index))
        if governor_index < len(words):
            governor = words[governor_index]
            governor_list.append(governor.pos.replace('$', '') + '_' + name)
        else:
            governor_list.append(NONE_DEPENDENCY + '_' + name)
        return governor_index, governor_list

    # add governor
    governor_index, governor_list = _get_governor(index, 'governor')
    pos_list.extend(governor_list)
    if governor_index < len(words) and int(words[governor_index].governor) != 0:
        _, ancestor_list = _get_governor(governor_index, 'ancestor')
        pos_list.extend(ancestor_list)

    def _get_children(_index, name):
        children = []
        child_list = []
        roots = [(i, w) for i, w in enumerate(words) if w.dependency_relation == 'root']
        start_index = 0
        end_index = len(words)
        for i, w in roots:
            if i <= _index:
                start_index = i
            else:
                end_index = i - 1
                break
        for i, w in enumerate(words[start_index:end_index]):
            if int(w.governor) == int(words[_index].index):
                children.append(start_index + i)
                child_list.append(w.pos.replace('$', '') + '_' + name)
        return children, child_list

    # add child
    children, child_list = _get_children(index, 'child')
    pos_list.extend(child_list)
    for i in children:
        grandchildren, grandchild_list = _get_children(i, 'grandchild')
        pos_list.extend(grandchild_list)
    return pos_list


def _vectorise_bag_of_pos_with_dependency(words, indexes):
    """Return pos list surrounding index
    Args:
        words (list): stanfordnlp word list object having pos attributes.
        indexes (List[int]): target indexes
    Return:
        pos_list (List[str]): xpo format string list
    """
    matrixes = []
    for i, index in enumerate(indexes):
        poss = _get_bag_of_pos_with_dependency(words, index)
        matrixes.append(" ".join(poss))
    return cv_dependencies.transform(matrixes).toarray().flatten()


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


def _get_sexial_labels(df):
    """Return sexila labels

    if Pronoun is she or her, return 0
    else return 1
    Args:
        df (DataFram): pandas DataFrame object
    Return:
        labels (array): label values. type of numpy int32 array. shaep is (N, 1)
    """
    labels = np.ones((len(df), 1), dtype=dtype)
    labels[df['Pronoun'].str.lower().isin(['she', 'her'])] = 0
    return labels


def _load_data(df, use_preprocessdata=False, save_path=None):
    """Load preprocess task speccific data.
    Args:
        df (DataFrame): target pandas DataFrame object.
        use_preprocessdata (bool): Wheter or not to use local preprocess file loading
        save_path (str): local preprocess file path
    Return:
        data (List[tuple]): words and indexes tuple list. Tulpe foramt is (sentence_words, [Pronnoun, A, B])
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
    return data


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
    data = _load_data(df, use_preprocessdata, save_path)
    X = []
    X2 = []
    for i, (words, indexes) in enumerate(data):
        X.append(
            _vectorise_bag_of_pos_with_position(words, indexes, DEFAULT_WINDOW_SIZE,
                                                targets=[df['Pronoun'][i], df['A'][i], df['B'][i]]))
        X2.append(_vectorise_bag_of_pos_with_dependency(words, indexes))

    X = np.array(X)
    X2 = np.array(X2)
    featur_len = int(X.shape[1] / 3)
    featur_len2 = int(X2.shape[1] / 3)
    X = np.concatenate((
        X[:, 0:featur_len] - X[:, featur_len:featur_len*2],
        X[:, 0:featur_len] - X[:, featur_len*2:featur_len*3],
        X[:, 0:featur_len] * X[:, featur_len:featur_len*2],
        X[:, 0:featur_len] * X[:, featur_len*2:featur_len*3],
        X2[:, 0:featur_len2] - X2[:, featur_len2:featur_len2*2],
        X2[:, 0:featur_len2] - X2[:, featur_len2*2:featur_len2*3],
        X2[:, 0:featur_len2] * X2[:, featur_len2:featur_len2*2],
        X2[:, 0:featur_len2] * X2[:, featur_len2*2:featur_len2*3],
        _get_sexial_labels(df),
        (df['Pronoun-offset'] - df['A-offset']).values.reshape(len(X), 1),
        (df['Pronoun-offset'] - df['B-offset']).values.reshape(len(X), 1)), axis=1)
    Y = _get_classify_labels(df)
    return X, Y


def train(use_preprocessdata=True):
    df = pandas.read_csv('dataset/gap-test.tsv', sep='\t')
    X, Y = _preprocess_data(df, use_preprocessdata=use_preprocessdata, save_path='preprocess_traindata.pkl')
    validation_df = pandas.read_csv('dataset/gap-validation.tsv', sep='\t')
    validation_X, validation_Y = _preprocess_data(validation_df, use_preprocessdata=use_preprocessdata, save_path='preprocess_valdata.pkl')

    def objective(trial):
        eta = trial.suggest_loguniform('eta', 0.001, 0.1)
        max_depth = trial.suggest_int('max_depth', 3, 25)
        gamma = trial.suggest_loguniform('gamma', 0.05, 1.0)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 7)
        subsample = trial.suggest_discrete_uniform('subsample', 0.6, 1.0, 0.1)
        colsample_bytree = trial.suggest_discrete_uniform('colsample_bytree', 0.6, 1.0, 0.1)
        model = xgb.XGBClassifier(
            max_depth=max_depth,
            eta=eta,
            gamma=gamma,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            n_jobs=1,
            random_state=0)
        # model = LogisticRegression(random_state=0)
        # model = SVC(C=10, probability=True, random_state=0)
        # model = MLPClassifier(hidden_layer_sizes=(50, 30, 30, 50), activation='relu', solver='adam', batch_size=128, random_state=0)

        def _log_loss(y_pred, y):
            """For XGBoost logloss calculator."""
            y = y.get_label()
            y_pred = y_pred.reshape((len(y), 3))
            return 'logloss', log_loss(y, y_pred)

        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, 'validation_0-logloss')
        model.fit(
            X,
            Y.flatten(),
            eval_set=[(validation_X, validation_Y.flatten())],
            eval_metric=_log_loss,
            callbacks=[pruning_callback],
            verbose=False)
        return log_loss(validation_Y, model.predict_proba(validation_X))

    study = optuna.create_study(
        study_name='gap-conf-kaggle',
        pruner=optuna.pruners.MedianPruner(),
        sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=100, n_jobs=-1)
    print("Best Params", study.best_params)
    print("Best Validation Value", study.best_value)

    model = xgb.XGBClassifier(n_jobs=-1, random_state=0, **study.best_params)
    model.fit(X, Y.flatten())
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    y_pred = model.predict(X)
    print("Train Accuracy:", accuracy_score(Y, y_pred))


def evaluate(test_data, use_preprocessdata=True):
    # train()
    X, Y = _preprocess_data(test_data, use_preprocessdata=use_preprocessdata, save_path='preprocess_testdata.pkl')
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    y_pred = model.predict(X)
    print("Test Accuracy:", accuracy_score(Y, y_pred))

    predicts = model.predict_proba(X)
    out_df = pandas.DataFrame(data=predicts, columns=['A', 'B', 'NEITHER'])
    out_df['ID'] = test_data['ID']
    return out_df

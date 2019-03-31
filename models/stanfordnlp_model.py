import pandas
import utils
import gpt2_estimator
import bert_estimator
import pickle
from collections import namedtuple
import itertools
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import optuna
import xgboost as xgb


dtype = np.int32
DEFAULT_NGRAM_WINDOW = 2
DEFAULT_WINDOW_SIZE = 5
NONE_DEPENDENCY = 'NONE'
FEATURE_TYPE = 'DEPENDENCY'
FEATURES = utils.POS_TYPES
if FEATURE_TYPE == 'UPOS':
    FEATURES = utils.UPOS_TYPES
elif FEATURE_TYPE == 'DEPENDENCY':
    FEATURES = utils.CONTENT_DEPRELS


DummyWord = namedtuple("DummyWord", "pos upos dependency_relation")
cv_normal = CountVectorizer(dtype=dtype)
cv_normal.fit(FEATURES + [utils.BEGIN_OF_SENTENCE, utils.END_OF_SENTENCE])
cv_ngram = CountVectorizer(dtype=dtype)
cv_ngram.fit(["_".join(p) for p in itertools.product(
    FEATURES + [utils.BEGIN_OF_SENTENCE, utils.END_OF_SENTENCE],
    repeat=DEFAULT_NGRAM_WINDOW)])
cv_position = CountVectorizer(token_pattern=r'\b[-\w][-\w]+\b', dtype=dtype)
cv_position.fit([p[0] + "_" + str(p[1]) for p in itertools.product(
    FEATURES + [utils.BEGIN_OF_SENTENCE, utils.END_OF_SENTENCE], range(-DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_SIZE+1))])
cv_upos_position = CountVectorizer(token_pattern=r'\b[-\w][-\w]+\b', dtype=dtype)
cv_upos_position.fit([p[0] + "_" + str(p[1]) for p in itertools.product(
    utils.UPOS_TYPES + [utils.BEGIN_OF_SENTENCE, utils.END_OF_SENTENCE], range(-DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_SIZE+1))])
dependencies = ['governor', 'child', 'ancestor', 'grandchild', 'sibling', 'sibling_child']
cv_dependencies = CountVectorizer(token_pattern=r'\b[-\w][-\w]+\b', dtype=dtype)
cv_dependencies.fit([p[0] + "_" + str(p[1]) for p in itertools.product(
    FEATURES + [NONE_DEPENDENCY], dependencies)])


def _get_word_feature(w):
    if FEATURE_TYPE == 'XPOS':
        return w.pos.replace('$', '')
    elif FEATURE_TYPE == 'UPOS':
        return w.upos
    elif FEATURE_TYPE == 'DEPENDENCY':
        return w.dependency_relation.split(':')[0]


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
    bos = DummyWord(pos=utils.BEGIN_OF_SENTENCE, upos=utils.BEGIN_OF_SENTENCE, dependency_relation=utils.BEGIN_OF_SENTENCE)
    eos = DummyWord(pos=utils.END_OF_SENTENCE, upos=utils.END_OF_SENTENCE, dependency_relation=utils.END_OF_SENTENCE)
    words = [bos] * N + words + [eos] * N
    index += N
    return [_get_word_feature(w) for w in words[index-N:index] + [words[index]] + words[index+target_len:index+target_len+N]]


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
    bos = DummyWord(pos=utils.BEGIN_OF_SENTENCE, upos=utils.BEGIN_OF_SENTENCE, dependency_relation=utils.BEGIN_OF_SENTENCE)
    eos = DummyWord(pos=utils.END_OF_SENTENCE, upos=utils.END_OF_SENTENCE, dependency_relation=utils.END_OF_SENTENCE)
    words = [bos] * (window_size + N) + words + [eos] * (window_size + N)
    index += (window_size + N)
    return [
        "_".join([_get_word_feature(w) for w in words[i:i+N]])
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
    bos = DummyWord(pos=utils.BEGIN_OF_SENTENCE, upos=utils.BEGIN_OF_SENTENCE, dependency_relation=utils.BEGIN_OF_SENTENCE)
    eos = DummyWord(pos=utils.END_OF_SENTENCE, upos=utils.END_OF_SENTENCE, dependency_relation=utils.END_OF_SENTENCE)
    words = [bos] * N + words + [eos] * N
    index += N
    return [_get_word_feature(w) + '_' + str(i-N) for i, w in enumerate(
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
    bos = DummyWord(pos=utils.BEGIN_OF_SENTENCE, upos=utils.BEGIN_OF_SENTENCE, dependency_relation=utils.BEGIN_OF_SENTENCE)
    eos = DummyWord(pos=utils.END_OF_SENTENCE, upos=utils.END_OF_SENTENCE, dependency_relation=utils.END_OF_SENTENCE)
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
        if int(words[_index].governor) == 0:
            # case _index word has no governer
            return -1, governor_list
        governor_index = _index + (int(words[_index].governor) - int(words[_index].index))
        if governor_index < len(words):
            governor = words[governor_index]
            governor_list.append(_get_word_feature(governor) + '_' + name)
        else:
            governor_list.append(NONE_DEPENDENCY + '_' + name)
        return governor_index, governor_list

    def _get_children(_index, name):
        children = []
        child_list = []
        roots = [(i, w) for i, w in enumerate(words) if int(w.index) == 1]
        start_index = 0
        end_index = len(words) - 1
        for i, w in roots:
            if i <= _index:
                start_index = i
            else:
                end_index = i - 1
                break
        for i, w in enumerate(words[start_index:end_index + 1]):
            if int(w.governor) == int(words[_index].index):
                children.append(start_index + i)
                child_list.append(_get_word_feature(w) + '_' + name)
        return children, child_list

    # add governor
    governor_index, governor_list = _get_governor(index, 'governor')
    if 0 <= governor_index < len(words):
        # case index word has a governer
        pos_list.extend(governor_list)
        if int(words[governor_index].governor) != 0:
            # case _index word has a governer
            # add ancestor
            _, ancestor_list = _get_governor(governor_index, 'ancestor')
            pos_list.extend(ancestor_list)

        # add sibling
        siblings, sibling_list = _get_children(governor_index, 'sibling')
        i_index = siblings.index(index)
        del sibling_list[i_index]
        del siblings[i_index]
        pos_list.extend(sibling_list)

        # add sibling list
        for i in siblings:
            sibling_children, sibling_child_list = _get_children(i, 'sibling_child')
            pos_list.extend(sibling_child_list)

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


def _get_governor(words, index):
    if int(words[index].governor) == 0:
        # case index word has no governer
        return DummyWord("", "", "")
    governor_index = index + (int(words[index].governor) - int(words[index].index))
    if governor_index < len(words):
        governor = words[governor_index]
        return governor
    else:
        return DummyWord("", "", "")


def _get_dependency_labels(words, indexes, targets):
    """Return dependency parser features

    Args:
        words (list): stanfordnlp word list object having pos attributes.
        indexes (List[int]): target indexes
        targets (List[str]): target word list. length is same as indexes
    Return:
        feature_list (List[int]): features list. shape is (2, 1)
    """
    pronounce_bop = _get_bag_of_pos_with_position(words, indexes[0], DEFAULT_WINDOW_SIZE, target_len=len(targets[0]))
    A_bop = _get_bag_of_pos_with_position(words, indexes[1], DEFAULT_WINDOW_SIZE, target_len=len(targets[1]))
    B_bop = _get_bag_of_pos_with_position(words, indexes[2], DEFAULT_WINDOW_SIZE, target_len=len(targets[2]))

    pronounce_governor = _get_governor(words, indexes[0])
    A_governor = _get_governor(words, indexes[1])
    B_governor = _get_governor(words, indexes[2])

    feature_list = [
        words[indexes[0]].dependency_relation.split(":")[0] == words[indexes[1]].dependency_relation.split(":")[0],
        words[indexes[0]].dependency_relation.split(":")[0] == words[indexes[2]].dependency_relation.split(":")[0],
        pronounce_governor.dependency_relation.split(":")[0] == A_governor.dependency_relation.split(":")[0],
        pronounce_governor.dependency_relation.split(":")[0] == B_governor.dependency_relation.split(":")[0],
        pronounce_governor.upos == A_governor.upos,
        pronounce_governor.upos == B_governor.upos,
        len(set(pronounce_bop) & set(A_bop)),
        len(set(pronounce_bop) & set(B_bop)),
        abs(indexes[0] - indexes[1]),
        abs(indexes[0] - indexes[2]),
        int(gpt2_estimator._check_pronounce_is_possessive(words, indexes[0]))
    ]
    return feature_list


def _get_same_sentence_features(words, indexes):
    """Return in same setence feature flags

    Args:
        words (list): stanfordnlp word list object having pos attributes.
        indexes (List[int]): target indexes
    Return:
        feature_list (List[int]): features list. shape is (2, 1)
    """
    roots = [(i, w) for i, w in enumerate(words) if int(w.index) == 1]
    points = [0, 0, 0]
    for i, w in roots:
        for j in range(len(indexes)):
            if i <= indexes[j]:
                points[0] += 1
    feature_list = [
        points[0] == points[1],
        points[0] == points[2],
        points[1] == points[2],
    ]
    return feature_list


def _get_gpt2_likelihood(words, indexes):
    """Return choice likelihoods

    Args:
        words (List[Words]): stanfordnlp word object list.
        indexe (List[int]): target index list. format is [Pronoun, A, B]
    Return:
        rates (List[float]): selection likelihood. [A, B, NEITHER]
    """
    woman = ["she", "her"]
    sentence = gpt2_estimator._get_scope_sentence(words, np.array(indexes))
    gender = "her" if words[indexes[0]].text.lower() in woman else "his"
    predicts = gpt2_estimator.predict(sentence + "\nQ: What's " + gender + " name?\nA:")

    A_rate1 = 0.00
    B_rate1 = 0.00

    for pairs in reversed(predicts):
        if words[indexes[1]].text.startswith(pairs[0]) and pairs[1] >= 0.2:
            A_rate1 = pairs[1]
        if words[indexes[2]].text.startswith(pairs[0]) and pairs[1] >= 0.2:
            B_rate1 = pairs[1]

    before_pronounce_sentence = gpt2_estimator._get_before_pronounce_sentence(words, indexes[0])
    predicts = gpt2_estimator.predict(sentence + " " + before_pronounce_sentence)
    A_rate2 = 0.00
    B_rate2 = 0.00

    for pairs in reversed(predicts):
        if words[indexes[1]].text.startswith(pairs[0]) and pairs[1] >= 0.2:
            A_rate2 = pairs[1]
        if words[indexes[2]].text.startswith(pairs[0]) and pairs[1] >= 0.2:
            B_rate2 = pairs[1]

    return [A_rate1, B_rate1, A_rate2, B_rate2]


def _bert_attentions(df, data):
    texts = [df['Text'][i] for i in range(len(df))]
    words = [[words[j].text.replace('`', '') for j in indexes] for words, indexes in data]
    indexes = [indexes for words, indexes in data]
    return bert_estimator._get_token_attentions(texts, words, indexes)


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
            words, pronnoun_index = utils.charpos_to_word_index(df['Text'][i], df['Pronoun-offset'][i], df['Pronoun'][i].split()[0])
            _, A_index = utils.charpos_to_word_index(df['Text'][i], df['A-offset'][i], df['A'][i].split()[0], words=words)
            _, B_index = utils.charpos_to_word_index(df['Text'][i], df['B-offset'][i], df['B'][i].split()[0], words=words)
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
    X3 = []
    X4 = []
    for i, (words, indexes) in enumerate(data):
        X.append(
            _vectorise_bag_of_pos_with_position(words, indexes, DEFAULT_WINDOW_SIZE,
                                                targets=[df['Pronoun'][i], df['A'][i], df['B'][i]]))
        X2.append(_vectorise_bag_of_pos_with_dependency(words, indexes))
        X3.append(_get_dependency_labels(words, indexes, targets=[df['Pronoun'][i], df['A'][i], df['B'][i]]))
        X4.append(_get_gpt2_likelihood(words, indexes))

    X5 = _bert_attentions(df, data)
    X5 = np.array(X5)

    X = np.array(X)
    X2 = np.array(X2)
    featur_len = int(X.shape[1] / 3)
    featur_len2 = int(X2.shape[1] / 3)
    X_pr = X[:, 0:featur_len]
    X_a = X[:, featur_len:featur_len*2]
    X_b = X[:, featur_len*2:featur_len*3]
    X2_pr = X2[:, 0:featur_len2]
    X2_a = X2[:, featur_len2:featur_len2*2]
    X2_b = X2[:, featur_len2*2:featur_len2*3]
    X = np.concatenate((
        X_pr - X_a,
        X_pr - X_b,
        X_pr * X_a,
        X_pr * X_b,
        X2_pr - X2_a,
        X2_pr - X2_b,
        X2_pr * X2_a,
        X2_pr * X2_b,
        X3,
        X5,
        (df['Pronoun-offset'] - df['A-offset']).values.reshape(len(X), 1),
        (df['Pronoun-offset'] - df['B-offset']).values.reshape(len(X), 1)
    ), axis=1)
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
    model.fit(
        np.concatenate([X, validation_X]),
        np.concatenate([Y, validation_Y]).flatten())
    with open('stanfordnlp_model.pkl', 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    y_pred = model.predict(X)
    print("Train Accuracy:", accuracy_score(Y, y_pred))


def evaluate(test_data, use_preprocessdata=True):
    gpt2_estimator.build()
    bert_estimator.build()
    train()
    X, Y = _preprocess_data(test_data, use_preprocessdata=use_preprocessdata, save_path='preprocess_testdata.pkl')
    with open('stanfordnlp_model.pkl', 'rb') as f:
        model = pickle.load(f)
    y_pred = model.predict(X)
    print("Test Accuracy:", accuracy_score(Y, y_pred))

    print("Confusion Matrix:\n", confusion_matrix(Y, y_pred))
    non_neithers = ((Y.flatten() != 2) & (y_pred != 2))
    print("Non Neithers Counts", sum(non_neithers))
    print("Non Neithers Test Accuracy:", accuracy_score(Y[non_neithers], y_pred[non_neithers]))

    predicts = model.predict_proba(X)
    out_df = pandas.DataFrame(data=predicts, columns=['A', 'B', 'NEITHER'])
    out_df['ID'] = test_data['ID']
    return out_df

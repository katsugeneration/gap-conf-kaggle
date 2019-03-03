import pandas
import utils
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
import optuna
import xgboost as xgb
from models import stanfordnlp_model


dtype = np.int32
DEFAULT_NGRAM_WINDOW = 2
DEFAULT_WINDOW_SIZE = 10
NONE_DEPENDENCY = 'NONE'


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


def _get_classify_labels(df):
    """Return task classify label

    format is following.
        data1 A is True or not
        ...
        data1 B is True or not
        ...
    Args:
        df (DataFram): pandas DataFrame object
    Return:
        labels (array): label values. type of numpy int32 array. shaep is (2*N, 1)
    """
    labels_A = np.zeros((len(df), 1), dtype=dtype)
    labels_A[df['A-coref']] = 1
    labels_B = np.zeros((len(df), 1), dtype=dtype)
    labels_B[df['B-coref']] = 1
    labels = np.concatenate([labels_A, labels_B])
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
    data = _load_data(df, use_preprocessdata, save_path)
    X = []
    X2 = []
    for i, (words, indexes) in enumerate(data):
        X.append(
            stanfordnlp_model._vectorise_bag_of_pos_with_position(words, indexes, DEFAULT_WINDOW_SIZE,
                                                                  targets=[df['Pronoun'][i], df['A'][i], df['B'][i]]))
        X2.append(stanfordnlp_model._vectorise_bag_of_pos_with_dependency(words, indexes))

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
    X_A = np.concatenate((
        X_pr,
        X_a,
        X2_pr,
        X2_a,
        X_pr - X_a,
        X_pr * X_a,
        X2_pr - X2_a,
        X2_pr * X2_a,
        stanfordnlp_model._get_sexial_labels(df),
        (df['Pronoun-offset'] - df['A-offset']).values.reshape(len(X), 1)), axis=1)
    X_B = np.concatenate((
        X_pr,
        X_b,
        X2_pr,
        X2_b,
        X_pr - X_b,
        X_pr * X_b,
        X2_pr - X2_b,
        X2_pr * X2_b,
        stanfordnlp_model._get_sexial_labels(df),
        (df['Pronoun-offset'] - df['B-offset']).values.reshape(len(X), 1)), axis=1)
    X = np.concatenate((X_A, X_B))
    Y = _get_classify_labels(df)
    return X, Y


def calculate_rate(y_pred):
    """Return categorical probability rate

    dimesnsion 0 is A likelihood
    dimesnsion 1 is B likelihood
    dimesnsion 2 is not A or B likelihood
    Args:
        y_pred (array): prediction probability array folloeing format.
            data1 A probability.
            ...
            data2 B probability.
            ...
    Return:
        labels (array): label values. type of numpy int32 array. shaep is (N, 1)
    """
    if len(y_pred.shape) == 2:
        y_pred = y_pred[:, 1]
    length = int(len(y_pred) / 2)
    y_pred_A = y_pred[:length]
    y_pred_B = y_pred[length:]

    result = np.concatenate([
        (y_pred_A).reshape(length, 1),
        (y_pred_B).reshape(length, 1),
        ((1 - y_pred_A) * (1 - y_pred_B)).reshape(length, 1),
    ], axis=1)
    result /= np.sum(result, axis=1, keepdims=1)
    return result


def train(use_preprocessdata=True):
    df = pandas.read_csv('dataset/gap-test.tsv', sep='\t')
    X, Y = _preprocess_data(df, use_preprocessdata=use_preprocessdata, save_path='preprocess_traindata.pkl')
    Y_labels = stanfordnlp_model._get_classify_labels(df)
    validation_df = pandas.read_csv('dataset/gap-validation.tsv', sep='\t')
    validation_X, validation_Y = _preprocess_data(validation_df, use_preprocessdata=use_preprocessdata, save_path='preprocess_valdata.pkl')
    validation_Y_labels = stanfordnlp_model._get_classify_labels(validation_df)

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

        def _log_loss(y_pred, y):
            """For XGBoost logloss calculator."""
            y_pred = calculate_rate(y_pred)
            return 'logloss', log_loss(validation_Y_labels, y_pred)

        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, 'validation_0-logloss')
        model.fit(
            X,
            Y.flatten(),
            eval_set=[(validation_X, validation_Y.flatten())],
            eval_metric=_log_loss,
            callbacks=[pruning_callback],
            verbose=False)
        return log_loss(validation_Y_labels, calculate_rate(model.predict_proba(validation_X)))

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
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    y_pred = calculate_rate(model.predict_proba(X))
    print("Train Accuracy:", accuracy_score(Y_labels, np.argmax(y_pred, axis=1)))


def evaluate(test_data, use_preprocessdata=True):
    train()
    X, Y = _preprocess_data(test_data, use_preprocessdata=use_preprocessdata, save_path='preprocess_testdata.pkl')
    Y_labels = stanfordnlp_model._get_classify_labels(test_data)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    pred = model.predict_proba(X)
    y_pred = calculate_rate(pred)
    print("Test Accuracy:", accuracy_score(Y_labels, np.argmax(y_pred, axis=1)))
    a = (Y_labels.flatten()[:20] != np.argmax(y_pred[:20], axis=1))
    print("Error Case", Y_labels[:20][a])
    print("Error Case Label", np.argmax(y_pred[:20][a], axis=1))
    print("Error Case Rate", y_pred[:20][a])
    print("A predictions", pred[:20][a])
    print("B predictions", pred[int(len(pred)/2):int(len(pred)/2)+20][a])

    data = _load_data(test_data, True, 'preprocess_testdata.pkl')
    for i, (words, indexes) in enumerate(data):
        if i in np.where(a == True)[0]:
            print("Index", i)
            print("Pronounce position", stanfordnlp_model._get_bag_of_pos_with_position(words, indexes[0], DEFAULT_WINDOW_SIZE, target_len=len(test_data['Pronoun'][i].split())))
            print("A position", stanfordnlp_model._get_bag_of_pos_with_position(words, indexes[1], DEFAULT_WINDOW_SIZE, target_len=len(test_data['A'][i].split())))
            print("B position", stanfordnlp_model._get_bag_of_pos_with_position(words, indexes[2], DEFAULT_WINDOW_SIZE, target_len=len(test_data['B'][i].split())))
            print("Pronounce dependency", stanfordnlp_model._get_bag_of_pos_with_dependency(words, indexes[0]))
            print("A dependency", stanfordnlp_model._get_bag_of_pos_with_dependency(words, indexes[1]))
            print("B dependency", stanfordnlp_model._get_bag_of_pos_with_dependency(words, indexes[2]))

    predicts = calculate_rate(model.predict_proba(X))
    out_df = pandas.DataFrame(data=predicts, columns=['A', 'B', 'NEITHER'])
    out_df['ID'] = test_data['ID']
    return out_df

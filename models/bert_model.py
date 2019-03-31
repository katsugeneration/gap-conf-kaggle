# Limitations under the MIT License.
# Copyright 2019 Katsuya Shimabukuro.

import pickle
import bert_estimator
import pandas
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from models import stanfordnlp_model
import optuna
import xgboost as xgb


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
    data = stanfordnlp_model._load_data(df, use_preprocessdata, save_path)
    X = []

    texts = [df['Text'][i] for i in range(len(df))]
    words = [[words[j].text.replace('`', '') for j in indexes] for words, indexes in data]
    indexes = [indexes for words, indexes in data]
    X = bert_estimator._get_token_attentions(texts, words, indexes)
    X = np.array(X)
    print(X.shape)
    Y = stanfordnlp_model._get_classify_labels(df)
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
    with open('bert_model.pkl', 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    y_pred = model.predict(X)
    print("Train Accuracy:", accuracy_score(Y, y_pred))


def evaluate(test_data, use_preprocessdata=True):
    bert_estimator.build()
    train()
    X, Y = _preprocess_data(test_data, use_preprocessdata=use_preprocessdata, save_path='preprocess_testdata.pkl')
    with open('bert_model.pkl', 'rb') as f:
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

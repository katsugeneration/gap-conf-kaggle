import os
import json
import random
import optuna
import pandas
import numpy as np
import tensorflow as tf
import bert_estimator
import gpt2_estimator
from models import stanfordnlp_model


POS_WITH_POSITION_SIZE = len(stanfordnlp_model.cv_position.vocabulary_)
POS_WITH_DEP_SIZE = len(stanfordnlp_model.cv_dependencies.vocabulary_)
BERT_VECTOR_SIZE = 768 * 2
SEED = 1


class MLP(tf.keras.Model):
    def __init__(self,
                 hidden_dims=100,
                 hidden_num=10,
                 last_dims=10,
                 drop_rate=0.5,
                 l1_weight=0.01,
                 l2_weight=0.01,
                 **kwargs):
        super(MLP, self).__init__()
        self.hidden_num = hidden_num
        self.drop_rate = drop_rate
        self.dense1 = tf.keras.layers.Dense(
            hidden_dims,
            use_bias=True)
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.hidden_layers = []
        self.hidden_bns = []
        self.hidden_dropouts = []
        for i in range(self.hidden_num):
            self.hidden_layers.append(tf.keras.layers.Dense(
                hidden_dims,
                use_bias=True))
            self.hidden_bns.append(tf.keras.layers.BatchNormalization())

        self.dense4 = tf.keras.layers.Dense(
            last_dims,
            activation=tf.nn.softmax,
            use_bias=True)

    def call(self, inputs):
        x = self.dense1(inputs)
        # x = self.bn1(x)
        x = tf.nn.tanh(x)
        x = tf.keras.layers.Dropout(self.drop_rate)(x)
        for i in range(self.hidden_num):
            old = x
            x = self.hidden_layers[i](x)
            # x = self.hidden_bns[i](x)
            x = tf.nn.tanh(x) + old
            x = tf.keras.layers.Dropout(self.drop_rate)(x)
        return self.dense4(x)


class ScoreRanker(tf.keras.Model):
    def __init__(self,
                 hidden_dims=150,
                 drop_rate=0.2,
                 l1_weight=0.01,
                 l2_weight=0.01,
                 emb_dims=30,
                 **kwargs):
        super(ScoreRanker, self).__init__()
        self.drop_rate = drop_rate
        self.dense1 = tf.keras.layers.Dense(
            hidden_dims,
            use_bias=True,
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_weight, l2=l2_weight))
        self.dense2 = tf.keras.layers.Dense(
            emb_dims,
            use_bias=True,
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_weight, l2=l2_weight))
        self.dropout = tf.keras.layers.Dropout(self.drop_rate)
        self.out = tf.keras.layers.Dense(1)

    def call(self, inputs):
        start_bert = (POS_WITH_POSITION_SIZE + POS_WITH_DEP_SIZE) * 3
        start_dep = POS_WITH_POSITION_SIZE * 3

        bert_p = inputs[:, start_bert:start_bert + BERT_VECTOR_SIZE]
        bert_a = inputs[:, start_bert + BERT_VECTOR_SIZE:start_bert + BERT_VECTOR_SIZE*2]
        bert_b = inputs[:, start_bert + BERT_VECTOR_SIZE*2:start_bert + BERT_VECTOR_SIZE*3]

        dep_p = inputs[:, start_dep:start_dep + POS_WITH_DEP_SIZE]
        dep_a = inputs[:, start_dep + POS_WITH_DEP_SIZE:start_dep + POS_WITH_DEP_SIZE * 2]
        dep_b = inputs[:, start_dep + POS_WITH_DEP_SIZE * 2:start_dep + POS_WITH_DEP_SIZE * 3]

        pa = tf.concat([bert_p, bert_a, bert_p * bert_a], -1)
        pb = tf.concat([bert_p, bert_b, bert_p * bert_b], -1)

        pa = self.dense1(pa)
        dep_pa = self.dense2(tf.concat([dep_p, dep_a, dep_p * dep_a], -1))
        pa = tf.nn.relu(tf.concat([pa, dep_pa], -1))
        pa = self.dropout(pa)
        pa_score = self.out(pa)

        pb = self.dense1(pb)
        dep_pb = self.dense2(tf.concat([dep_p, dep_b, dep_p * dep_b], -1))
        pb = tf.nn.relu(tf.concat([pb, dep_pb], -1))
        pb = self.dropout(pb)
        pb_score = self.out(pb)

        outputs = tf.nn.softmax(tf.concat([pa_score, pb_score, tf.zeros_like(pa_score)], -1))
        return outputs


def set_seed():
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    session_conf = tf.ConfigProto()
    tf.set_random_seed(SEED)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    tf.keras.backend.set_session(sess)
    return sess


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
    X2 = []
    for i, (words, indexes) in enumerate(data):
        X.append(
            stanfordnlp_model._vectorise_bag_of_pos_with_position(words, indexes, stanfordnlp_model.DEFAULT_WINDOW_SIZE,
                                                                  targets=[df['Pronoun'][i], df['A'][i], df['B'][i]]))
        X2.append(stanfordnlp_model._vectorise_bag_of_pos_with_dependency(words, indexes))
    X5 = bert_estimator.embed_by_bert(df)
    X5 = np.array(X5)
    X = np.concatenate([
        X, X2, X5
    ], axis=-1)
    Y = stanfordnlp_model._get_classify_labels(df)
    return X, Y


def train(use_preprocessdata=True):
    df = pandas.read_csv('dataset/gap-test.tsv', sep='\t')
    X, Y = _preprocess_data(df, use_preprocessdata=use_preprocessdata, save_path='preprocess_traindata.pkl')
    validation_df = pandas.read_csv('dataset/gap-validation.tsv', sep='\t')
    validation_X, validation_Y = _preprocess_data(validation_df, use_preprocessdata=use_preprocessdata, save_path='preprocess_valdata.pkl')

    def objective(trial):
        hidden_dims = trial.suggest_int('hidden_dims', 100, 300)
        drop_rate = trial.suggest_uniform('drop_rate', 0.1, 1.0)
        l1_weight = trial.suggest_loguniform('l1_weight', 0.001, 0.1)
        l2_weight = trial.suggest_loguniform('l2_weight', 0.001, 0.1)
        emb_dims = trial.suggest_int('emb_dims', 10, 50)
        lr = 0.001
        batch_size = 128

        def _on_epoch_end(epoch, logs=None):
            """Call for pruning when end epoch.

            Args:
                epoch (int): number of epoch count.
                logs (dict): epoch metrics.
            """
            intermediate_value = logs['val_loss']
            trial.report(intermediate_value, epoch)

            # Handle pruning based on the intermediate value.
            if epoch > 10 and trial.should_prune(epoch):
                print(epoch, logs)
                tf.keras.backend.clear_session()
                raise optuna.structs.TrialPruned()

        def _on_train_end(logs=None):
            """Call for printing train data metrics when end training.

            Args:
                logs (dict): training metrics.
            """
            evals = model.evaluate(
                X,
                tf.keras.utils.to_categorical(Y, num_classes=3),
                verbose=0)
            print('Train Metrics:', evals)
            evals = model.evaluate(
                validation_X,
                tf.keras.utils.to_categorical(validation_Y, num_classes=3),
                verbose=0)
            print('Eval Metrics:', evals)

        set_seed()
        model = ScoreRanker(hidden_dims, drop_rate, l1_weight, l2_weight, emb_dims)
        model.compile(
            tf.keras.optimizers.Adam(lr),
            loss='categorical_crossentropy',
            metrics=['acc', 'categorical_crossentropy']
        )

        early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=100)
        model.fit(
            x=X,
            y=tf.keras.utils.to_categorical(Y, num_classes=3),
            batch_size=batch_size,
            epochs=1000,
            workers=4,
            validation_data=(validation_X, tf.keras.utils.to_categorical(validation_Y, num_classes=3)),
            callbacks=[tf.keras.callbacks.LambdaCallback(on_train_end=_on_train_end), early_stop],
            verbose=0)

        evals = model.evaluate(
            validation_X,
            tf.keras.utils.to_categorical(validation_Y, num_classes=3),
            verbose=0)
        tf.keras.backend.clear_session()
        return evals[2]

    study = optuna.create_study(
        study_name='gap-conf-kaggle',
        pruner=optuna.pruners.MedianPruner(),
        sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=20, n_jobs=1)
    print("Best Params", study.best_params)
    print("Best Validation Value", study.best_value)

    set_seed()
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=100)
    model = ScoreRanker(**study.best_params)
    model.compile(
        tf.keras.optimizers.Adam(0.001),
        loss='categorical_crossentropy',
        metrics=['acc', 'categorical_crossentropy']
    )
    model.fit(
        x=X,
        y=tf.keras.utils.to_categorical(Y, num_classes=3),
        batch_size=128,
        epochs=1000,
        workers=4,
        validation_data=(validation_X, tf.keras.utils.to_categorical(validation_Y, num_classes=3)),
        callbacks=[early_stop],
        verbose=0)

    model.save_weights('model.h5')
    with open('model.json', 'w') as f:
        json.dump(study.best_params, f)
    evals = model.evaluate(
            X,
            tf.keras.utils.to_categorical(Y, num_classes=3),
            verbose=0)
    print("Train Accuracy:", evals[1])
    tf.keras.backend.clear_session()


def evaluate(test_data, use_preprocessdata=True):
    # gpt2_estimator.build()
    bert_estimator.build()
    train()
    X, Y = _preprocess_data(test_data, use_preprocessdata=use_preprocessdata, save_path='preprocess_testdata.pkl')

    set_seed()
    with open('model.json', 'r') as f:
        params = json.load(f)
    model = ScoreRanker(**params)
    model.compile(
        tf.keras.optimizers.Adam(0.001),
        loss='categorical_crossentropy',
        metrics=['acc', 'categorical_crossentropy']
    )
    model.build((None, X.shape[1]))
    model.load_weights('model.h5')

    evals = model.evaluate(
            X,
            tf.keras.utils.to_categorical(Y, num_classes=3),
            verbose=0)
    print("Test Accuracy:", evals[1])

    predicts = model.predict(X)
    out_df = pandas.DataFrame(data=predicts, columns=['A', 'B', 'NEITHER'])
    out_df['ID'] = test_data['ID']
    return out_df

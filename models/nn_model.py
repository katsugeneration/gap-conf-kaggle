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


def set_seed():
    os.environ['PYTHONHASHSEED'] = '0'
    random.seed(0)
    np.random.seed(0)
    session_conf = tf.ConfigProto()
    tf.set_random_seed(0)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    tf.keras.backend.set_session(sess)
    return sess


def train(use_preprocessdata=True):
    df = pandas.read_csv('dataset/gap-test.tsv', sep='\t')
    X, Y = stanfordnlp_model._preprocess_data(df, use_preprocessdata=use_preprocessdata, save_path='preprocess_traindata.pkl')
    validation_df = pandas.read_csv('dataset/gap-validation.tsv', sep='\t')
    validation_X, validation_Y = stanfordnlp_model._preprocess_data(validation_df, use_preprocessdata=use_preprocessdata, save_path='preprocess_valdata.pkl')

    def objective(trial):
        hidden_dims = trial.suggest_int('hidden_dims', 100, 300)
        hidden_num = trial.suggest_int('hidden_num', 1, 10)
        drop_rate = trial.suggest_uniform('drop_rate', 0.1, 1.0)
        # l1_weight = 0.0  # trial.suggest_loguniform('l1_weight', 0.0001, 0.01)
        # l2_weight = trial.suggest_loguniform('l2_weight', 0.0001, 0.01)
        lr = trial.suggest_loguniform('lr', 0.001, 0.1)
        batch_size = trial.suggest_int('batch_size', 32, 128)
        # epochs = trial.suggest_int('epochs', 50, 200)

        def _on_epoch_end(epoch, logs=None):
            """Call for pruning when end epoch.

            Args:
                epoch (int): number of epoch count.
                logs (dict): epoch metrics.
            """
            intermediate_value = 2 * (1 - logs['val_acc']) + logs['val_loss']
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
        model = MLP(hidden_dims, hidden_num, 3, drop_rate)
        model.compile(
            tf.keras.optimizers.Adam(lr),
            loss='categorical_crossentropy',
            metrics=['acc', 'categorical_crossentropy']
        )

        model.fit(
            x=X,
            y=tf.keras.utils.to_categorical(Y, num_classes=3),
            batch_size=batch_size,
            epochs=1000,
            workers=4,
            validation_data=(validation_X, tf.keras.utils.to_categorical(validation_Y, num_classes=3)),
            callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=_on_epoch_end, on_train_end=_on_train_end)],
            verbose=0)

        evals = model.evaluate(
            validation_X,
            tf.keras.utils.to_categorical(validation_Y, num_classes=3),
            verbose=0)
        tf.keras.backend.clear_session()
        return 2 * (1 - evals[1]) + evals[2]

    study = optuna.create_study(
        study_name='gap-conf-kaggle',
        pruner=optuna.pruners.MedianPruner(),
        sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=100, n_jobs=1)
    print("Best Params", study.best_params)
    print("Best Validation Value", study.best_value)

    set_seed()
    model = MLP(last_dims=3, **study.best_params)
    model.compile(
        tf.keras.optimizers.Adam(lr=study.best_params['lr']),
        loss='categorical_crossentropy',
        metrics=['acc', 'categorical_crossentropy']
    )
    model.fit(
        np.concatenate([X, validation_X]),
        tf.keras.utils.to_categorical(np.concatenate([Y, validation_Y]), num_classes=3),
        batch_size=study.best_params['batch_size'],
        epochs=1000,
        workers=4,
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
    gpt2_estimator.build()
    bert_estimator.build()
    train()
    X, Y = stanfordnlp_model._preprocess_data(test_data, use_preprocessdata=use_preprocessdata, save_path='preprocess_testdata.pkl')

    set_seed()
    with open('model.json', 'r') as f:
        params = json.load(f)
    model = MLP(last_dims=3, **params)
    model.compile(
        tf.keras.optimizers.Adam(lr=params['lr']),
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

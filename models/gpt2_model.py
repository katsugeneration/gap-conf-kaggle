# Limitations under the MIT License.
# Copyright 2019 Katsuya Shimabukuro.

import os
import re
import json
import pandas
import numpy as np
import tensorflow as tf
import importlib.util
from sklearn.metrics import accuracy_score, log_loss
from models import stanfordnlp_model

# Load Open AI GPT-2 module
gpt2_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "gpt-2")
spec = importlib.util.spec_from_file_location("model", os.path.join(gpt2_path, "src", "model.py"))
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)
spec = importlib.util.spec_from_file_location("encoder", os.path.join(gpt2_path, "src", "encoder.py"))
encoder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(encoder)

model_name = '117M'
sess = None
enc = encoder.get_encoder(model_name)
targets = None
context = None
rates = None


def build():
    """Build model."""
    global sess
    global targets
    global rates
    global context

    hparams = model.default_hparams()
    with open(os.path.join(gpt2_path, 'models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    context = tf.placeholder(tf.int32, [1, None])
    lm_output = model.model(hparams=hparams, X=context, past=None, reuse=tf.AUTO_REUSE)
    logits = lm_output['logits'][:, :, :hparams.n_vocab]
    values, indices = tf.nn.top_k(logits[:, -1, :], k=10)
    rates = tf.math.softmax(values)
    indices = tf.reshape(indices, [10, 1])

    sess = tf.Session()
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(os.path.join(gpt2_path, 'models', model_name))
    saver.restore(sess, ckpt)

    targets = indices
    rates = rates
    context = context


def predict(text):
    """Predict next word.

    Args:
        text (str): input text for predction
    Return:
        predictions (List[Tuple[str, int]]): preidcted word and probability list.
    """
    context_tokens = enc.encode(text)
    words, probabilities = sess.run([targets, rates], {context: [context_tokens]})
    return [(enc.decode(o).strip(), p) for o, p in zip(words, probabilities[0])]


def _get_scope_sentence(words, indexes):
    """Return target sentence

    Args:
        words (List[Words]): stanfordnlp word object list.
        indexe (List[int]): target index list. format is [Pronoun, A, B]
    Return:
        sentence (str): target word contains sentence.
    """
    roots = [(i, w) for i, w in enumerate(words) if int(w.index) == 1]
    start_index = 0
    end_index = len(words) - 1
    for i, w in roots:
        if (i <= indexes).all():
            start_index = i
        elif (i >= indexes).all():
            end_index = i - 1
            break
    sentence = " ".join([w.text for w in words[start_index:end_index+1]])
    sentence = re.sub(r" (\W)", r"\1", sentence)
    sentence = re.sub(r" n't", r"n't", sentence)
    return sentence


def calcurate_likelihood(words, indexes, y):
    """Return choice likelihoods

    Args:
        words (List[Words]): stanfordnlp word object list.
        indexe (List[int]): target index list. format is [Pronoun, A, B]
    Return:
        rates (List[float]): selection likelihood. [A, B, NEITHER]
    """
    woman = ["she", "her"]
    sentence = _get_scope_sentence(words, np.array(indexes))
    gender = "her" if words[indexes[0]].text.lower() in woman else "his"
    predicts = predict(sentence + "\nQ: What's " + gender + " name?\nA:")

    A_rate = 0.00
    B_rate = 0.00

    for pairs in reversed(predicts):
        if words[indexes[1]].text.startswith(pairs[0]) and pairs[1] >= 0.2:
            A_rate = pairs[1]
        if words[indexes[2]].text.startswith(pairs[0]) and pairs[1] >= 0.2:
            B_rate = pairs[1]

    if A_rate == 0.0 and B_rate == 0.0:
        rates = np.array([0.2, 0.2, 0.6], np.float32)
    else:
        rates = np.array([A_rate, B_rate, 0], np.float32)
    return rates / np.sum(rates)


def _get_governor(words, index):
    governor_list = []
    if int(words[index].governor) == 0:
        # case index word has no governer
        return -1, governor_list
    governor_index = index + (int(words[index].governor) - int(words[index].index))
    if governor_index < len(words):
        governor = words[governor_index]
        return stanfordnlp_model._get_word_feature(governor)
    else:
        return stanfordnlp_model.NONE_DEPENDENCY


def calculate_syntax_likelihood(words, indexes):
    """Return choice likelihoods

    Args:
        words (List[Words]): stanfordnlp word object list.
        indexes (List[int]): target index list. format is [Pronoun, A, B]
    Return:
        rates (List[float]): selection likelihood. [A, B, NEITHER]
    """
    pronounce_dependency_relation = words[indexes[0]].dependency_relation
    A_dependency_relation = words[indexes[1]].dependency_relation
    B_dependency_relation = words[indexes[2]].dependency_relation
    pronounce_governor = _get_governor(words, indexes[0])
    A_governor = _get_governor(words, indexes[1])
    B_governor = _get_governor(words, indexes[2])

    if pronounce_dependency_relation == A_dependency_relation and pronounce_governor == A_governor:
        rates = np.array([1.0, 0.0, 0.0], np.float32)
    elif pronounce_dependency_relation == B_dependency_relation and pronounce_governor == B_governor:
        rates = np.array([0.0, 1.0, 0.0], np.float32)
    else:
        rates = np.array([0.2, 0.2, 0.6], np.float32)
    return rates / np.sum(rates)


def evaluate(test_data, use_preprocessdata=True):
    build()
    validation_df = pandas.read_csv('dataset/gap-validation.tsv', sep='\t')
    data = stanfordnlp_model._load_data(validation_df, use_preprocessdata, 'preprocess_valdata.pkl')
    Y = stanfordnlp_model._get_classify_labels(validation_df)
    predicts = np.ndarray([len(validation_df), 3], dtype=np.float32)
    # data = stanfordnlp_model._load_data(test_data, use_preprocessdata, 'preprocess_testdata.pkl')
    # Y = stanfordnlp_model._get_classify_labels(test_data)
    # predicts = np.ndarray([len(test_data), 3], dtype=np.float32)
    for i, (words, indexes) in enumerate(data):
        # predicts[i] = calculate_syntax_likelihood(words, indexes)
        predicts[i] = calcurate_likelihood(words, indexes, Y[i])
        if np.argmax(predicts[i]) == 2:
            predicts[i] = calculate_syntax_likelihood(words, indexes)
            # predicts[i] = calcurate_likelihood(words, indexes, Y[i])

    print("A predict", sum(np.argmax(predicts, axis=1) == 0))
    print("B predict", sum(np.argmax(predicts, axis=1) == 1))
    print("Non predict", sum(np.argmax(predicts, axis=1) == 2))
    print("Test Accuracy:", accuracy_score(Y, np.argmax(predicts, axis=1)))

    non_neithers = (2 != np.argmax(predicts, axis=1))
    print("Non Neithers Test Accuracy:", accuracy_score(Y[non_neithers], np.argmax(predicts[non_neithers], axis=1)))

    corrects = (Y.flatten() == np.argmax(predicts, axis=1))
    print("Correct loss", log_loss(Y[corrects], predicts[corrects]))
    print("Loss", log_loss(Y, predicts))

    out_df = pandas.DataFrame(data=predicts, columns=['A', 'B', 'NEITHER'])
    out_df['ID'] = test_data['ID']
    return out_df
# Limitations under the MIT License.
# Copyright 2019 Katsuya Shimabukuro.

import os
import re
import json
import tensorflow as tf
import importlib.util

# Load Open AI GPT-2 module
gpt2_path = os.path.join(os.path.dirname(__file__), "gpt-2")
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
    values, indices = tf.nn.top_k(logits[:, -1, :], k=5)
    rates = tf.math.softmax(values)
    indices = tf.reshape(indices, [5, 1])

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


def _get_before_pronounce_sentence(words, index):
    """Return before pronounce sentence

    Args:
        words (List[Words]): stanfordnlp word object list.
        index (int): Pronounce index.
    Return:
        sentence (str): target word contains sentence to pronounce.
    """
    roots = [(i, w) for i, w in enumerate(words) if int(w.index) == 1]
    start_index = 0
    for i, w in roots:
        if i <= index:
            start_index = i
        else:
            break
    governor_index = index + (int(words[index].governor) - int(words[index].index))
    if governor_index < index:
        start_index = governor_index
    sentence = " ".join([w.text for w in words[start_index:index]])
    sentence = re.sub(r" (\W)", r"\1", sentence)
    sentence = re.sub(r" n't", r"n't", sentence)
    return sentence


def _check_pronounce_is_possessive(words, index):
    """whether pronoun is Possessive or not.

    Args:
        words (List[Words]): stanfordnlp word object list.
        index (int): Pronounce index.
    Return:
        is_possessive (bool): pronoun is Possessive flag.
    """
    pronoun = words[index]
    return (pronoun.xpos == 'PRP$')

# Limitations under the MIT License.
# Copyright 2019 Katsuya Shimabukuro.

import os
import json
import numpy as np
import tensorflow as tf
import importlib.util

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


def evaluate(test_data):
    predict()

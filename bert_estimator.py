# Limitations under the MIT License.
# Copyright 2019 Katsuya Shimabukuro.
# Refer to https://github.com/google-research/bert/blob/master/extract_features.py

import os
import numpy as np
import tensorflow as tf
import importlib.util

# Load BERT module
gpt2_path = os.path.join(os.path.dirname(__file__), "bert")
spec = importlib.util.spec_from_file_location("modeling", os.path.join(gpt2_path, "modeling.py"))
modeling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modeling)
spec = importlib.util.spec_from_file_location("tokenization", os.path.join(gpt2_path, "tokenization.py"))
tokenization = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tokenization)

BERT_PATH = "multi_cased_L-12_H-768_A-12/"
seq_length = 512
estimator = None
tokenizer = None


class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []

    for example in examples:
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


def read_examples(texts):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    for text in texts:
        line = tokenization.convert_to_unicode(text)
        line = line.strip()
        examples.append(
            InputExample(unique_id=0, text_a=line, text_b=None))
    return examples


def model_fn_builder(bert_config, init_checkpoint, use_tpu, use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        input_type_ids = features["input_type_ids"]

        model = modeling.BertModel(
            config=bert_config,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=input_type_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        if mode != tf.estimator.ModeKeys.PREDICT:
            raise ValueError("Only PREDICT modes are supported: %s" % (mode))

        tvars = tf.trainable_variables()
        scaffold_fn = None
        (assignment_map,
            initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
            tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        all_attention_heads = model.get_all_attention_heads()

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode, predictions=tf.convert_to_tensor(all_attention_heads),
            scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def input_fn_builder(features, seq_length):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_unique_ids = []
    all_input_ids = []
    all_input_mask = []
    all_input_type_ids = []

    for feature in features:
        all_unique_ids.append(feature.unique_id)
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_input_type_ids.append(feature.input_type_ids)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "unique_ids":
                tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_type_ids":
                tf.constant(
                    all_input_type_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
        })

        d = d.batch(batch_size=batch_size, drop_remainder=False)
        return d

    return input_fn


def build():
    global estimator
    global tokenizer

    # Initialize BERT settings
    bert_config = modeling.BertConfig.from_json_file(BERT_PATH + "bert_config.json")
    tokenizer = tokenization.FullTokenizer(
        vocab_file=BERT_PATH + "vocab.txt", do_lower_case=True)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=BERT_PATH + "bert_model.ckpt",
        use_tpu=False,
        use_one_hot_embeddings=False)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        master=None,
        tpu_config=tf.contrib.tpu.TPUConfig(
            num_shards=8,
            per_host_input_for_training=is_per_host))

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        predict_batch_size=10)


def predict(text):
    """Return attention values.

    Args:
        text (str): input text for predction
    Return:
        attention_values (List[Float]): attention values. shape is (num_layers, targets_len, targets_len, num_heads)
        tokens (List(str))
    """
    examples = read_examples(text)
    features = convert_examples_to_features(
        examples=examples, seq_length=seq_length, tokenizer=tokenizer)

    input_fn = input_fn_builder(
        features=features, seq_length=seq_length)

    predictions = estimator.predict(input_fn, yield_single_examples=True)
    return predictions, [f.tokens for f in features]


def _get_token_attentions(text, words, indexes):
    """Return target attention values

    Args:
        text (str): input text for predction
        words (List[Words]): stanfordnlp word object list.
        indexe (List[int]): target index list. format is [Pronoun, A, B]
    Return:
        attentions ([List[Float]]): word to word attention values. shape is (num_layers, len(words), len(words), num_heads)
    """
    targets = [sorted(enumerate(zip(w, indices)), key=lambda x: x[1][1]) for w, indices in zip(words, indexes)]
    attention_values, tokens = predict(text)

    all_attentions = []
    for k, avar in enumerate(attention_values):
        before = 0
        word_indeces = [0] * len(words[k])
        for i, (word, index) in targets[k]:
            indices = [j for j, x in enumerate(tokens[k]) if word.lower().startswith(x) and j > before]
            before = min(indices, key=lambda x: abs(x - index))
            word_indeces[i] = before

        attentions = []
        for i in word_indeces:
            attentions.append(avar[:, word_indeces, i:i+1])
        attentions = np.transpose(attentions, (3, 1, 2, 0, 4))[0].flatten()
        all_attentions.append(attentions)
    return all_attentions


# build()
# print(_get_token_attentions("Zoe Telford -- played the police officer girlfriend of Simon, Maggie. Dumped by Simon in the final episode of series 1, after he slept with Jenny, and is not seen again. Phoebe Thomas played Cheryl Cassidy, Pauline's friend and also a year 11 pupil in Simon's class. Dumped her boyfriend following Simon's advice after he wouldn't have sex with her but later realised this was due to him catching crabs off her friend Pauline.",
# ['her', 'Cheryl', 'Pauline'], [57, 39, 42]))

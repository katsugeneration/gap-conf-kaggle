import pandas
import numpy as np
import utils
from models import stanfordnlp_model
from nose.tools import eq_, ok_


def test_get_bag_of_pos():
    words, index = utils.charpos_to_word_index(
            "Zoe Telford -- played the police officer girlfriend of Simon, Maggie. Dumped by Simon in the final episode of series 1, after he slept with Jenny, and is not seen again. Phoebe Thomas played Cheryl Cassidy, Pauline's friend and also a year 11 pupil in Simon's class. Dumped her boyfriend following Simon's advice after he wouldn't have sex with her but later realised this was due to him catching crabs off her friend Pauline.",
            274)
    poss = stanfordnlp_model._get_bag_of_pos(words, index, 5)
    eq_(11, len(poss))
    eq_([w.pos.replace('$', '') for w in words[index-5:index+6]], poss)


def test_get_bag_of_pos_case_start_0():
    words, index = utils.charpos_to_word_index(
            "Zoe Telford -- played the police officer girlfriend of Simon, Maggie. Dumped by Simon in the final episode of series 1, after he slept with Jenny, and is not seen again. Phoebe Thomas played Cheryl Cassidy, Pauline's friend and also a year 11 pupil in Simon's class. Dumped her boyfriend following Simon's advice after he wouldn't have sex with her but later realised this was due to him catching crabs off her friend Pauline.",
            274)
    eq_(11, len(stanfordnlp_model._get_bag_of_pos(words, 0, 5)))


def test_vectorise_bag_of_pos():
    words, index = utils.charpos_to_word_index(
            "Zoe Telford -- played the police officer girlfriend of Simon, Maggie. Dumped by Simon in the final episode of series 1, after he slept with Jenny, and is not seen again. Phoebe Thomas played Cheryl Cassidy, Pauline's friend and also a year 11 pupil in Simon's class. Dumped her boyfriend following Simon's advice after he wouldn't have sex with her but later realised this was due to him catching crabs off her friend Pauline.",
            274)
    vectors = stanfordnlp_model._vectorise_bag_of_pos(words, [index, index], 5)
    eq_(11 * 2, np.sum(vectors))
    eq_((2 * 36, ), vectors.shape)


def test_get_classify_labels():
    data = pandas.DataFrame(data=[[True, False], [False, True], [False, False]], columns=['A-coref', 'B-coref'])
    labels = stanfordnlp_model._get_classify_labels(data)
    np.testing.assert_array_equal([[0], [1], [2]], labels)


def test_get_bag_of_pos_ngram():
    words, index = utils.charpos_to_word_index(
            "Zoe Telford -- played the police officer girlfriend of Simon, Maggie. Dumped by Simon in the final episode of series 1, after he slept with Jenny, and is not seen again. Phoebe Thomas played Cheryl Cassidy, Pauline's friend and also a year 11 pupil in Simon's class. Dumped her boyfriend following Simon's advice after he wouldn't have sex with her but later realised this was due to him catching crabs off her friend Pauline.",
            274)
    poss = stanfordnlp_model._get_bag_of_pos_ngram(words, index, 5, 2)
    eq_(11, len(poss))
    eq_([words[i].pos.replace('$', '') + "_" + words[i+1].pos.replace('$', '') for i in range(index-5, index+6)], poss)


def test_get_bag_of_pos_ngram_case_start_0():
    words, index = utils.charpos_to_word_index(
            "Zoe Telford -- played the police officer girlfriend of Simon, Maggie. Dumped by Simon in the final episode of series 1, after he slept with Jenny, and is not seen again. Phoebe Thomas played Cheryl Cassidy, Pauline's friend and also a year 11 pupil in Simon's class. Dumped her boyfriend following Simon's advice after he wouldn't have sex with her but later realised this was due to him catching crabs off her friend Pauline.",
            274)
    eq_(11, len(stanfordnlp_model._get_bag_of_pos_ngram(words, 0, 5, 2)))


def test_vectorise_bag_of_pos_ngram():
    words, index = utils.charpos_to_word_index(
            "Zoe Telford -- played the police officer girlfriend of Simon, Maggie. Dumped by Simon in the final episode of series 1, after he slept with Jenny, and is not seen again. Phoebe Thomas played Cheryl Cassidy, Pauline's friend and also a year 11 pupil in Simon's class. Dumped her boyfriend following Simon's advice after he wouldn't have sex with her but later realised this was due to him catching crabs off her friend Pauline.",
            274)
    vectors = stanfordnlp_model._vectorise_bag_of_pos_ngram(words, [index, index], 5)
    eq_(11 * 2, np.sum(vectors))
    eq_((2 * 36 * 36, ), vectors.shape)


def test_get_bag_of_pos_with_position():
    words, index = utils.charpos_to_word_index(
            "Zoe Telford -- played the police officer girlfriend of Simon, Maggie. Dumped by Simon in the final episode of series 1, after he slept with Jenny, and is not seen again. Phoebe Thomas played Cheryl Cassidy, Pauline's friend and also a year 11 pupil in Simon's class. Dumped her boyfriend following Simon's advice after he wouldn't have sex with her but later realised this was due to him catching crabs off her friend Pauline.",
            274)
    poss = stanfordnlp_model._get_bag_of_pos_with_position(words, index, 5)
    eq_(11, len(poss))
    eq_([w.pos.replace('$', '')  + '_' + str(i-5) for i, w in enumerate(words[index-5:index+6])], poss)


def test_vectorise_bag_of_pos_with_position():
    words, index = utils.charpos_to_word_index(
            "Zoe Telford -- played the police officer girlfriend of Simon, Maggie. Dumped by Simon in the final episode of series 1, after he slept with Jenny, and is not seen again. Phoebe Thomas played Cheryl Cassidy, Pauline's friend and also a year 11 pupil in Simon's class. Dumped her boyfriend following Simon's advice after he wouldn't have sex with her but later realised this was due to him catching crabs off her friend Pauline.",
            274)
    vectors = stanfordnlp_model._vectorise_bag_of_pos_with_position(words, [index, index], 3)
    eq_(7 * 2, np.sum(vectors))
    eq_((2 * 36 * (stanfordnlp_model.DEFAULT_WINDOW_SIZE * 2 + 1), ), vectors.shape)


def test_get_sexial_labels():
        df = pandas.DataFrame(data=['he', 'she', 'her', 'His', 'him', 'She', 'He', 'Her'], columns=['Pronoun'])
        labels = stanfordnlp_model._get_sexial_labels(df)
        np.testing.assert_array_equal([1, 0, 0, 1, 1, 0, 1, 0], labels.flatten())

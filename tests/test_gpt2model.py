import pandas
import numpy as np
import utils
import gpt2_estimator
from models import gpt2_model
from nose.tools import eq_, ok_


def test_get_predictions():
    gpt2_estimator.build()
    predicts = gpt2_estimator.predict("Phoebe Thomas played Cheryl Cassidy, Pauline's friend and also a year 11 pupil in Simon's class. Dumped her boyfriend following Simon's advice after he wouldn't have sex with her but later realised this was due to him catching crabs off her friend Pauline.\nQ: What's her name?\nA:")
    eq_('Cheryl', predicts[0][0])
    eq_(np.float32, type(predicts[0][1]))


def test_get_scope_sentence():
    words, index1 = utils.charpos_to_word_index(
            "Zoe Telford -- played the police officer girlfriend of Simon, Maggie. Dumped by Simon in the final episode of series 1, after he slept with Jenny, and is not seen again. Phoebe Thomas played Cheryl Cassidy, Pauline's friend and also a year 11 pupil in Simon's class. Dumped her boyfriend following Simon's advice after he wouldn't have sex with her but later realised this was due to him catching crabs off her friend Pauline.",
            274, 'her')
    words, index2 = utils.charpos_to_word_index(
            "Zoe Telford -- played the police officer girlfriend of Simon, Maggie. Dumped by Simon in the final episode of series 1, after he slept with Jenny, and is not seen again. Phoebe Thomas played Cheryl Cassidy, Pauline's friend and also a year 11 pupil in Simon's class. Dumped her boyfriend following Simon's advice after he wouldn't have sex with her but later realised this was due to him catching crabs off her friend Pauline.",
            207, 'Pauline')
    sentence = gpt2_estimator._get_scope_sentence(words, np.array([index1, index2]))
    eq_("Phoebe Thomas played Cheryl Cassidy, Pauline's friend and also a year 11 pupil in Simon's class. Dumped her boyfriend following Simon's advice after he wouldn't have sex with her but later realised this was due to him catching crabs off her friend Pauline.", sentence)


def test_get_before_pronounce_sentence_case_true():
    words, index1 = utils.charpos_to_word_index(
            "Zoe Telford -- played the police officer girlfriend of Simon, Maggie. Dumped by Simon in the final episode of series 1, after he slept with Jenny, and is not seen again. Phoebe Thomas played Cheryl Cassidy, Pauline's friend and also a year 11 pupil in Simon's class. Dumped her boyfriend following Simon's advice after he wouldn't have sex with her but later realised this was due to him catching crabs off her friend Pauline.",
            274, 'her')
    is_possessive = gpt2_estimator._check_pronounce_is_possessive(words, index1)
    eq_(True, is_possessive)


def test_get_before_pronounce_sentence_case_false():
    words, index1 = utils.charpos_to_word_index(
            "Her initial ambition was to become a ``cultured'' private secretary working in the highest cultural circles. A month later, the Michigan League building opened, serving co-eds as the Michigan Union served men. Hartwig began by doing secretarial work there, when Dr. Margaret Bell, the head of women's physical education, spotted her.",
            329, 'her')
    is_possessive = gpt2_estimator._check_pronounce_is_possessive(words, index1)
    eq_(False, is_possessive)


def test_get_before_pronounce_sentence():
    words, index1 = utils.charpos_to_word_index(
            "Zoe Telford -- played the police officer girlfriend of Simon, Maggie. Dumped by Simon in the final episode of series 1, after he slept with Jenny, and is not seen again. Phoebe Thomas played Cheryl Cassidy, Pauline's friend and also a year 11 pupil in Simon's class. Dumped her boyfriend following Simon's advice after he wouldn't have sex with her but later realised this was due to him catching crabs off her friend Pauline.",
            274, 'her')
    sentence = gpt2_estimator._get_before_pronounce_sentence(words, index1)
    eq_("Dumped", sentence)


def test_calcurate_likelihood():
    words, index1 = utils.charpos_to_word_index(
            "Zoe Telford -- played the police officer girlfriend of Simon, Maggie. Dumped by Simon in the final episode of series 1, after he slept with Jenny, and is not seen again. Phoebe Thomas played Cheryl Cassidy, Pauline's friend and also a year 11 pupil in Simon's class. Dumped her boyfriend following Simon's advice after he wouldn't have sex with her but later realised this was due to him catching crabs off her friend Pauline.",
            274, 'her')
    words, index2 = utils.charpos_to_word_index(
            "Zoe Telford -- played the police officer girlfriend of Simon, Maggie. Dumped by Simon in the final episode of series 1, after he slept with Jenny, and is not seen again. Phoebe Thomas played Cheryl Cassidy, Pauline's friend and also a year 11 pupil in Simon's class. Dumped her boyfriend following Simon's advice after he wouldn't have sex with her but later realised this was due to him catching crabs off her friend Pauline.",
            191, 'Cheryl')
    words, index3 = utils.charpos_to_word_index(
            "Zoe Telford -- played the police officer girlfriend of Simon, Maggie. Dumped by Simon in the final episode of series 1, after he slept with Jenny, and is not seen again. Phoebe Thomas played Cheryl Cassidy, Pauline's friend and also a year 11 pupil in Simon's class. Dumped her boyfriend following Simon's advice after he wouldn't have sex with her but later realised this was due to him catching crabs off her friend Pauline.",
            207, 'Pauline')
    predicts = gpt2_model.calcurate_likelihood(words, np.array([index1, index2, index3]))
    ok_(predicts[0] > 0)
    ok_(predicts[1] == 0)
    ok_(predicts[2] == 0)

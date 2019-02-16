import utils
from models import stanfordnlp_model
from nose.tools import eq_, ok_


def test_get_bag_of_pos():
    words, index = utils.charpos_to_word_index(
            "Zoe Telford -- played the police officer girlfriend of Simon, Maggie. Dumped by Simon in the final episode of series 1, after he slept with Jenny, and is not seen again. Phoebe Thomas played Cheryl Cassidy, Pauline's friend and also a year 11 pupil in Simon's class. Dumped her boyfriend following Simon's advice after he wouldn't have sex with her but later realised this was due to him catching crabs off her friend Pauline.",
            274)
    poss = stanfordnlp_model._get_bag_of_pos(words, index, 5)
    eq_(11, len(poss))
    eq_([w.pos for w in words[index-5:index+6]], poss)


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
    eq_((2 * 36, ), vectors.shape)

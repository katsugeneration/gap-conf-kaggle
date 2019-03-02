import utils
from nose.tools import eq_, ok_


def test_charpos_to_word_index():
    words, index = utils.charpos_to_word_index(
        "Zoe Telford -- played the police officer girlfriend of Simon, Maggie. Dumped by Simon in the final episode of series 1, after he slept with Jenny, and is not seen again. Phoebe Thomas played Cheryl Cassidy, Pauline's friend and also a year 11 pupil in Simon's class. Dumped her boyfriend following Simon's advice after he wouldn't have sex with her but later realised this was due to him catching crabs off her friend Pauline.",
        274, 'her')
    eq_(words[index].text, 'her')

def test_charpos_to_word_index_case_name():
    words, index = utils.charpos_to_word_index(
        "Zoe Telford -- played the police officer girlfriend of Simon, Maggie. Dumped by Simon in the final episode of series 1, after he slept with Jenny, and is not seen again. Phoebe Thomas played Cheryl Cassidy, Pauline's friend and also a year 11 pupil in Simon's class. Dumped her boyfriend following Simon's advice after he wouldn't have sex with her but later realised this was due to him catching crabs off her friend Pauline.",
        207, 'Pauline')
    eq_(words[index].text, 'Pauline')

def test_charpos_to_word_index_case_name_two_parts():
    words, index = utils.charpos_to_word_index(
        "Zoe Telford -- played the police officer girlfriend of Simon, Maggie. Dumped by Simon in the final episode of series 1, after he slept with Jenny, and is not seen again. Phoebe Thomas played Cheryl Cassidy, Pauline's friend and also a year 11 pupil in Simon's class. Dumped her boyfriend following Simon's advice after he wouldn't have sex with her but later realised this was due to him catching crabs off her friend Pauline.",
        191, 'Cheryl')
    eq_(words[index].text, 'Cheryl')

def test_charpos_to_word_index_case_error_pattern_1():
    words, index = utils.charpos_to_word_index(
        "Geller is reckoned to have been among the best ten players in the world for around twenty years. He was awarded the International Master title in 1951, and the International Grandmaster title the following year. Geller played in 23 USSR Chess Championship s, a record equalled by Mark Taimanov, achieving good results in many. He won in 1955 at Moscow (URS-ch22) when, despite losing five games, he finished with 12/19, then defeated Smyslov in the playoff match by the score of +1 =6.",
        327, 'He')
    eq_(words[index].text, 'He')


def test_charpos_to_word_index_use_words():
    words, index = utils.charpos_to_word_index(
        "Zoe Telford -- played the police officer girlfriend of Simon, Maggie. Dumped by Simon in the final episode of series 1, after he slept with Jenny, and is not seen again. Phoebe Thomas played Cheryl Cassidy, Pauline's friend and also a year 11 pupil in Simon's class. Dumped her boyfriend following Simon's advice after he wouldn't have sex with her but later realised this was due to him catching crabs off her friend Pauline.",
        191, 'Cheryl')
    words, index = utils.charpos_to_word_index(
        "Zoe Telford -- played the police officer girlfriend of Simon, Maggie. Dumped by Simon in the final episode of series 1, after he slept with Jenny, and is not seen again. Phoebe Thomas played Cheryl Cassidy, Pauline's friend and also a year 11 pupil in Simon's class. Dumped her boyfriend following Simon's advice after he wouldn't have sex with her but later realised this was due to him catching crabs off her friend Pauline.",
        274, 'her', words=words)
    eq_(words[index].text, 'her')

def test_get_same_word_index():
    words, indexes = utils.get_same_word_index(
        "Zoe Telford -- played the police officer girlfriend of Simon, Maggie. Dumped by Simon in the final episode of series 1, after he slept with Jenny, and is not seen again. Phoebe Thomas played Cheryl Cassidy, Pauline's friend and also a year 11 pupil in Simon's class. Dumped her boyfriend following Simon's advice after he wouldn't have sex with her but later realised this was due to him catching crabs off her friend Pauline.",
        'her')
    eq_(3, len(indexes))
    for i in indexes:
        eq_("".join([w.text for w in words[i]]), 'her')

def test_get_same_word_index_case_change_form():
    words, indexes = utils.get_same_word_index(
        "He grew up in Evanston, Illinois the second oldest of five children including his brothers, Fred and Gordon and sisters, Marge (Peppy) and Marilyn. His high school days were spent at New Trier High School in Winnetka, Illinois. MacKenzie studied with Bernard Leach from 1949 to 1952. His simple, wheel-thrown functional pottery is heavily influenced by the oriental aesthetic of Shoji Hamada and Kanjiro Kawai.",
        'His')
    eq_(4, len(indexes))
    for i in indexes:
        eq_("".join([w.lemma for w in words[i]]), 'he')

def test_get_same_word_index_case_name():
    words, indexes = utils.get_same_word_index(
        "Zoe Telford -- played the police officer girlfriend of Simon, Maggie. Dumped by Simon in the final episode of series 1, after he slept with Jenny, and is not seen again. Phoebe Thomas played Cheryl Cassidy, Pauline's friend and also a year 11 pupil in Simon's class. Dumped her boyfriend following Simon's advice after he wouldn't have sex with her but later realised this was due to him catching crabs off her friend Pauline.",
        'Pauline')
    eq_(2, len(indexes))
    for i in indexes:
        eq_("".join([w.text for w in words[i]]), 'Pauline')

def test_get_same_word_index_case_name_two_parts():
    words, indexes = utils.get_same_word_index(
        "Zoe Telford -- played the police officer girlfriend of Simon, Maggie. Dumped by Simon in the final episode of series 1, after he slept with Jenny, and is not seen again. Phoebe Thomas played Cheryl Cassidy, Pauline's friend and also a year 11 pupil in Simon's class. Dumped her boyfriend following Simon's advice after he wouldn't have sex with her but later realised this was due to him catching crabs off her friend Pauline.",
        'Cheryl Cassidy')
    eq_(1, len(indexes))
    for i in indexes:
        eq_(" ".join([w.text for w in words[i]]), 'Cheryl Cassidy')

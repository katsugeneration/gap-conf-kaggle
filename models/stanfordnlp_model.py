import pandas
import utils
from collections import namedtuple


START_OF_SENTENCE = "SOS"
END_OF_SENTENCE = "EOS"


def _get_bag_of_pos(words, index, N):
    DummyWord = namedtuple("DummyWord", "pos")
    sos = DummyWord(pos=START_OF_SENTENCE)
    eos = DummyWord(pos=END_OF_SENTENCE)
    words = [sos] * N + words + [eos] * N
    index += N
    return [w.pos for w in words[index-N:index+N+1]]


def evaluate(test_data):
    df = pandas.read_csv('dataset/gap-development.tsv', sep='\t')
    utils.charpos_to_word_index(df['Text'], df['Pronoun-offset'])

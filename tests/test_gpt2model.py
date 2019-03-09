import pandas
import numpy as np
from models import gpt2_model
from nose.tools import eq_, ok_


def test_get_predictions():
    gpt2_model.build()
    predicts = gpt2_model.predict("Phoebe Thomas played Cheryl Cassidy, Pauline's friend and also a year 11 pupil in Simon's class. Dumped her boyfriend following Simon's advice after he wouldn't have sex with her but later realised this was due to him catching crabs off her friend Pauline.\nQ: What's her name?\nA:")
    eq_('Cheryl', predicts[0][0])
    eq_(np.float32, type(predicts[0][1]))

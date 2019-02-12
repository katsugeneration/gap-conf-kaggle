import score
from nose.tools import ok_


def test_calc_score():
    s = score.calc_score('dataset/gap-test.tsv', 'kaggle-data/sample_submission_stage_1.csv')
    ok_(abs(s - 1.09861) < 0.00001)

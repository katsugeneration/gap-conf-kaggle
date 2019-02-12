from sklearn import metrics
import numpy as np
import pandas


def calc_score(correct_file, predict_file):
    """Calculate loglos between correct answer and predict likelihood.

    Args:
        correct_file (str): correct file path formated to test data.
        predict_file (str): predict file path formated to submission data.
    """
    correct_df = pandas.read_csv(correct_file, sep='\t')
    predict_df = pandas.read_csv(predict_file)
    N = len(correct_df)

    y_true = np.zeros((N, 3))
    y_predict = np.zeros((N, 3))

    y_true[correct_df['A-coref'], 0] = 1
    y_true[correct_df['B-coref'], 1] = 1
    y_true[~(correct_df['A-coref'] | correct_df['B-coref']), 2] = 1
    y_predict[:, 0] = predict_df['A']
    y_predict[:, 1] = predict_df['B']
    y_predict[:, 2] = predict_df['NEITHER']
    return metrics.log_loss(y_true, y_predict)

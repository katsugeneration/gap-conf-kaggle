import pandas


def evaluate(test_data):
    evals = pandas.DataFrame()
    evals['ID'] = test_data['ID']
    evals['A'] = 0.33333
    evals['B'] = 0.33333
    evals['NEITHER'] = 0.33333
    return evals

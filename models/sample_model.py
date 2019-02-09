import pandas


def evaluate():
    test_data = pandas.read_csv('kaggle-data/test_stage_1.tsv', sep='\t')
    evals = pandas.DataFrame()
    evals['ID'] = test_data['ID']
    evals['A'] = 0.33333
    evals['B'] = 0.33333
    evals['NEITHER'] = 0.33333
    return evals

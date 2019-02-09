import pandas


def evaluate(test_data):
    df = pandas.read_csv('dataset/gap-development.tsv', sep='\t')
    she_list = ['her', 'She', 'she', 'Her', 'hers', 'Hers']
    he_list = ['him', 'He', 'he', 'Him', 'His', 'his']

    she_df = df[df['Pronoun'].isin(she_list)]
    she_names = pandas.concat([
        she_df['A'][she_df['A-coref']],
        she_df['B'][she_df['B-coref']]]).unique()
    he_df = df[df['Pronoun'].isin(he_list)]
    he_names = pandas.concat([
        he_df['A'][he_df['A-coref']],
        he_df['B'][he_df['B-coref']]]).unique()
    she_only = list(set(she_names) - set(he_names))
    he_only = list(set(he_names) - set(she_names))

    test_data = pandas.read_csv('kaggle-data/test_stage_1.tsv', sep='\t')
    evals = pandas.DataFrame()
    evals['ID'] = test_data['ID']
    evals['A'] = 0.0
    evals['B'] = 0.0
    evals['NEITHER'] = 0.0

    A_true = ((test_data['A'][test_data['Pronoun'].isin(she_list)].isin(she_only)) |
              (test_data['A'][test_data['Pronoun'].isin(he_list)].isin(he_only)))
    B_true = ((test_data['B'][test_data['Pronoun'].isin(she_list)].isin(she_only)) |
              (test_data['B'][test_data['Pronoun'].isin(he_list)].isin(he_only)))
    evals.loc[A_true, 'A'] = 1.0
    evals.loc[B_true, 'B'] = 1.0
    evals.loc[~(A_true | B_true), 'A'] = 0.33333
    evals.loc[~(A_true | B_true), 'B'] = 0.33333
    evals.loc[~(A_true | B_true), 'NEITHER'] = 0.33333
    return evals

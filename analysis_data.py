import pandas
import load_data
import numpy as np
from models import stanfordnlp_model


print('Train data Analysis')
df = load_data.load('dataset/gap-development.tsv')
print("pronoun after A count", (df['A-offset'] > df['Pronoun-offset']).sum(axis=0))
print("pronoun after B count", (df['B-offset'] > df['Pronoun-offset']).sum(axis=0))
print("A is True count", df['A-coref'].sum(axis=0))
print("B is True count", df['B-coref'].sum(axis=0))
print("non both count", ((df['A-coref'] == False) & (df['B-coref'] == False)).sum(axis=0))
print("pronoun after A AND A is True count", (df['A-coref'] & (df['A-offset'] > df['Pronoun-offset'])).sum(axis=0))
print("pronoun Bfter B BND B is True count", (df['B-coref'] & (df['B-offset'] > df['Pronoun-offset'])).sum(axis=0))

a_df = df['Pronoun-offset'] - df['A-offset']
b_df = df['Pronoun-offset'] - df['B-offset']
print("A is near pronoun than B AND A is True count", (a_df.abs() < b_df.abs())[df['A-coref']].sum(axis=0))
print("A is near pronoun than B AND B is True count", (a_df.abs() < b_df.abs())[df['B-coref']].sum(axis=0))
print("A is near pronoun than B", (a_df.abs() < b_df.abs()).sum(axis=0))

print("Pronoun unique values", df['Pronoun'].unique())
print("A unique values", df['A'].unique())
print("B unique values", df['B'].unique())

she_df = df[df['Pronoun'].isin(['her', 'She', 'she', 'Her'])]
she_names = pandas.concat([
    she_df['A'][she_df['A-coref']],
    she_df['B'][she_df['B-coref']]]).unique()
print("She values count", len(she_names))
he_df = df[df['Pronoun'].isin(['him', 'He', 'he', 'Him'])]
he_names = pandas.concat([
    he_df['A'][he_df['A-coref']],
    he_df['B'][he_df['B-coref']]]).unique()
print("He values count", len(he_names))
print("She unique values count", len(set(she_names) - set(he_names)))
print("He unique values count", len(set(he_names) - set(she_names)))

# Pos different check
data = stanfordnlp_model._load_data(df, True, 'preprocess_testdata.pkl')
X = []
for i, (words, indexes) in enumerate(data):
    X.append(
        stanfordnlp_model._vectorise_bag_of_pos_with_position(words, indexes, stanfordnlp_model.DEFAULT_WINDOW_SIZE,
                                                              targets=[df['Pronoun'][i], df['A'][i], df['B'][i]]))
X = np.array(X)
num = len(X)
featur_len = int(X.shape[1] / 3)
a_trues = X[df['A-coref']]
b_trues = X[df['B-coref']]
all_ngs = X[~(df['A-coref'] | df['B-coref'])]
true_diffs = np.concatenate([
    a_trues[:, 0:featur_len] - a_trues[:, featur_len:featur_len*2],
    b_trues[:, 0:featur_len] - b_trues[:, featur_len*2:featur_len*3],
])
false_diffs = np.concatenate([
    b_trues[:, 0:featur_len] - b_trues[:, featur_len:featur_len*2],
    a_trues[:, 0:featur_len] - a_trues[:, featur_len*2:featur_len*3],
    all_ngs[:, 0:featur_len] - all_ngs[:, featur_len:featur_len*2],
    all_ngs[:, 0:featur_len] - all_ngs[:, featur_len*2:featur_len*3]
])
print("True label pos diff mean", np.absolute(true_diffs).sum(axis=1).mean())
print("False label pos diff mean", np.absolute(false_diffs).sum(axis=1).mean())
print("True label pos diff variance", np.absolute(true_diffs).sum(axis=1).var())
print("False label pos diff variance", np.absolute(false_diffs).sum(axis=1).var())
print("Count True label is diff large case", (np.absolute(true_diffs).sum(axis=1) > np.absolute(false_diffs[:len(true_diffs)]).sum(axis=1)).sum(axis=0))

true_sames = np.concatenate([
    a_trues[:, 0:featur_len] * a_trues[:, featur_len:featur_len*2],
    b_trues[:, 0:featur_len] * b_trues[:, featur_len*2:featur_len*3],
])
false_sames = np.concatenate([
    b_trues[:, 0:featur_len] * b_trues[:, featur_len:featur_len*2],
    a_trues[:, 0:featur_len] * a_trues[:, featur_len*2:featur_len*3],
    all_ngs[:, 0:featur_len] * all_ngs[:, featur_len:featur_len*2],
    all_ngs[:, 0:featur_len] * all_ngs[:, featur_len*2:featur_len*3]
])
print("True label pos same mean", true_sames.sum(axis=1).mean())
print("False label pos same mean", false_sames.sum(axis=1).mean())
print("True label pos same variance", true_sames.sum(axis=1).var())
print("False label pos same variance", false_sames.sum(axis=1).var())
print("Count True label is same large case", (true_sames.sum(axis=1) > false_sames[:len(true_sames)].sum(axis=1)).sum(axis=0))

vocabulary = {v: k for k, v in stanfordnlp_model.cv_position.vocabulary_.items()}
true_feature_diffs = np.absolute(true_diffs).sum(axis=0)
false_feature_diffs = np.absolute(false_diffs).sum(axis=0)
print("True Lable differ feature")
for i in np.argsort(true_feature_diffs)[-10:]:
    if i in vocabulary:
        print(vocabulary[i], true_feature_diffs[i])
    else:
        print("Nothing", true_feature_diffs[i])
print("False Lable differ feature")
for i in np.argsort(false_feature_diffs)[-10:]:
    if i in vocabulary:
        print(vocabulary[i], false_feature_diffs[i])
    else:
        print("Nothing", false_feature_diffs[i])

positive_true_feature_diffs = true_diffs.clip(min=0).sum(axis=0)
positive_false_feature_diffs = false_diffs.clip(min=0).sum(axis=0)
print("Poitive True Lable differ feature")
for i in np.argsort(positive_true_feature_diffs)[-10:]:
    if i in vocabulary:
        print(vocabulary[i], positive_true_feature_diffs[i])
    else:
        print("Nothing", positive_true_feature_diffs[i])
print("Poitive False Lable differ feature")
for i in np.argsort(positive_false_feature_diffs)[-10:]:
    if i in vocabulary:
        print(vocabulary[i], positive_false_feature_diffs[i])
    else:
        print("Nothing", positive_false_feature_diffs[i])

negative_true_feature_diffs = (-true_diffs).clip(min=0).sum(axis=0)
negative_false_feature_diffs = (-false_diffs).clip(min=0).sum(axis=0)
print("Negative True Lable differ feature")
for i in np.argsort(negative_true_feature_diffs)[-10:]:
    if i in vocabulary:
        print(vocabulary[i], negative_true_feature_diffs[i])
    else:
        print("Nothing", negative_true_feature_diffs[i])
print("Negative False Lable differ feature")
for i in np.argsort(negative_false_feature_diffs)[-10:]:
    if i in vocabulary:
        print(vocabulary[i], negative_false_feature_diffs[i])
    else:
        print("Nothing", negative_false_feature_diffs[i])

true_feature_sames = np.absolute(true_sames).sum(axis=0)
false_feature_sames = np.absolute(false_sames).sum(axis=0)
print("True Lable more same feature")
for i in np.argsort(true_feature_sames)[-10:]:
    if i in vocabulary:
        print(vocabulary[i], true_feature_sames[i])
    else:
        print("Nothing", true_feature_sames[i])
print("False Lable more same feature")
for i in np.argsort(false_feature_sames)[-10:]:
    if i in vocabulary:
        print(vocabulary[i], false_feature_sames[i])
    else:
        print("Nothing", false_feature_sames[i])

print('Train data Analysis')
df = load_data.load('dataset/gap-test.tsv')
print("pronoun after A count", (df['A-offset'] > df['Pronoun-offset']).sum(axis=0))
print("pronoun after B count", (df['B-offset'] > df['Pronoun-offset']).sum(axis=0))
print("A is True count", df['A-coref'].sum(axis=0))
print("B is True count", df['B-coref'].sum(axis=0))
print("non both count", ((df['A-coref'] == False) & (df['B-coref'] == False)).sum(axis=0))
print("pronoun after A AND A is True count", (df['A-coref'] & (df['A-offset'] > df['Pronoun-offset'])).sum(axis=0))
print("pronoun Bfter B BND B is True count", (df['B-coref'] & (df['B-offset'] > df['Pronoun-offset'])).sum(axis=0))

a_df = df['Pronoun-offset'] - df['A-offset']
b_df = df['Pronoun-offset'] - df['B-offset']
print("A is near pronoun than B AND A is True count", (a_df.abs() < b_df.abs())[df['A-coref']].sum(axis=0))
print("A is near pronoun than B AND B is True count", (a_df.abs() < b_df.abs())[df['B-coref']].sum(axis=0))
print("A is near pronoun than B", (a_df.abs() < b_df.abs()).sum(axis=0))

print("Pronoun unique values", df['Pronoun'].unique())
print("A unique values", df['A'].unique())
print("B unique values", df['B'].unique())

she_df = df[df['Pronoun'].isin(['her', 'She', 'she', 'Her'])]
she_names = pandas.concat([
    she_df['A'][she_df['A-coref']],
    she_df['B'][she_df['B-coref']]]).unique()
print("She values count", len(she_names))
he_df = df[df['Pronoun'].isin(['him', 'He', 'he', 'Him'])]
he_names = pandas.concat([
    he_df['A'][he_df['A-coref']],
    he_df['B'][he_df['B-coref']]]).unique()
print("He values count", len(he_names))
print("She unique values count", len(set(she_names) - set(he_names)))
print("He unique values count", len(set(he_names) - set(she_names)))


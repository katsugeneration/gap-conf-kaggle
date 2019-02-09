import pandas
import load_data


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

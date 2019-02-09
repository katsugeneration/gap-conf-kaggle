import pandas


def load(filepath: str):
    df = pandas.read_csv(filepath, sep='\t')
    return df

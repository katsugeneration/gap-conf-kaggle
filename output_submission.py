import importlib
import argparse
import load_data
import score

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model',
    dest='model',
    required=True,
    help='model module name'
)
parser.add_argument(
    '--output',
    dest='output_path',
    required=True,
    help='model module name'
)
args = parser.parse_args()

model = importlib.import_module('models.' + args.model)
df = load_data.load('kaggle-data/test_stage_1.tsv')
evals = model.evaluate(df)
evals.to_csv(args.output_path, columns=['ID', 'A', 'B', 'NEITHER'], header=True, index=False)
# print(score.calc_score('dataset/gap-development.tsv', args.output_path))
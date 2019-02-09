import importlib
import argparse

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
evals = model.evaluate()
evals.to_csv(args.output_path, columns=['ID', 'A', 'B', 'NEITHER'], header=True, index=False)

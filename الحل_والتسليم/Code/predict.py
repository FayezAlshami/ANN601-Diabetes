"""Show a held-out dataset prediction using the validation-selected model."""
from pathlib import Path
import argparse
import json
import numpy as np
import joblib
from train import load_data, FEATURES

ROOT = Path(__file__).resolve().parent


def predict_record(record, artifact):
    values = np.array(record, dtype=float, copy=True).reshape(1, -1)
    if values.shape != (1, 8) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Provide eight finite non-negative values in the documented feature order')
    for j in artifact['zero_missing_indices']:
        if values[0, j] == 0:
            values[0, j] = np.nan
    p = float(artifact['pipeline'].predict_proba(values)[0, 1])
    return {'model': artifact['name'], 'probability_class_1': p,
            'predicted_class': int(p >= artifact['threshold'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-position', type=int, default=0)
    parser.add_argument('--values', type=float, nargs=8, metavar='VALUE', help=', '.join(FEATURES))
    args = parser.parse_args()
    selected = json.loads((ROOT / 'results/metrics.json').read_text())['selected_model']
    artifact = joblib.load(ROOT / 'results' / (selected + '.joblib'))
    if args.values is not None:
        result = predict_record(args.values, artifact)
    else:
        raw, _, y, _ = load_data()
        indices = np.load(ROOT / 'results/split_indices.npz')['test']
        if not 0 <= args.test_position < len(indices):
            parser.error(f'test-position must be between 0 and {len(indices)-1}')
        i = int(indices[args.test_position])
        result = {'row_index': i, 'true_class': int(y[i]), **predict_record(raw[i, :8], artifact)}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

"""Independently verify the saved splits, preprocessing, predictions and metrics."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix
from train import load_data
from predict import predict_record

ROOT = Path(__file__).resolve().parent


def main():
    m = json.loads((ROOT / 'results/metrics.json').read_text())
    raw, x, y, _ = load_data()
    assert hashlib.sha256((ROOT / 'data/diabetes.csv').read_bytes()).hexdigest() == m['dataset_sha256']
    splits = np.load(ROOT / 'results/split_indices.npz')
    train, val, test = [splits[k] for k in ['train', 'validation', 'test']]
    assert len(set(train) | set(val) | set(test)) == 768
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    for name in ['MLP_A', 'MLP_B']:
        artifact = joblib.load(ROOT / 'results' / (name + '.joblib'))
        pipeline = artifact['pipeline']
        np.testing.assert_allclose(pipeline['imputer'].statistics_, np.nanmedian(x[train], axis=0))
        imputed = pipeline['imputer'].transform(x[train])
        np.testing.assert_allclose(pipeline['scaler'].mean_, imputed.mean(axis=0))
        np.testing.assert_allclose(pipeline['scaler'].scale_, imputed.std(axis=0))
        probability = pipeline.predict_proba(x[test])[:, 1]
        pred = (probability >= .5).astype(int)
        saved = m['models'][name]['test']
        for key, function in [('accuracy', accuracy_score), ('precision', precision_score), ('recall', recall_score), ('f1', f1_score)]:
            np.testing.assert_allclose(saved[key], function(y[test], pred), rtol=0, atol=1e-14)
        np.testing.assert_allclose(saved['log_loss'], log_loss(y[test], probability), rtol=0, atol=1e-14)
        np.testing.assert_array_equal(saved['confusion_matrix'], confusion_matrix(y[test], pred))
        with (ROOT / 'results' / (name + '_test_predictions.csv')).open() as f:
            rows = list(csv.DictReader(f))
        np.testing.assert_array_equal(test, [int(r['row_index']) for r in rows])
        np.testing.assert_array_equal(pred, [int(r['predicted']) for r in rows])
        np.testing.assert_allclose(probability, [float(r['probability']) for r in rows])
        for j, i in enumerate(test):
            result = predict_record(raw[i, :8], artifact)
            np.testing.assert_allclose(result['probability_class_1'], probability[j], atol=1e-14)
        history = np.loadtxt(ROOT / 'results' / (name + '_history.csv'), delimiter=',', skiprows=1)
        best = int(history[np.argmin(history[:, 2]), 0])
        assert best == m['models'][name]['best_epoch']
        np.testing.assert_allclose(history[best - 1, 2], m['models'][name]['validation']['log_loss'])
    assert m['selected_model'] == min(m['models'], key=lambda n: m['models'][n]['validation']['log_loss'])
    print('PASS: dataset checksum, disjoint splits, training-only preprocessing, both models, all test predictions, metrics and validation selection')


if __name__ == '__main__':
    main()

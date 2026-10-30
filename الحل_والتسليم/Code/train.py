"""Compare the two neural networks specified in the approved ANN601 proposal."""
from pathlib import Path
from copy import deepcopy
import csv
import hashlib
import json
import platform
import numpy as np
import scipy
import sklearn
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, log_loss, confusion_matrix, roc_auc_score)
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
DATA = ROOT / 'data' / 'diabetes.csv'
OUT = ROOT / 'results'
SEED = 42
MAX_EPOCHS = 500
PATIENCE = 40
MIN_DELTA = 1e-4
FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
ZERO_MISSING = [1, 2, 3, 4, 5]
ARCHITECTURES = {'MLP_A': (32,), 'MLP_B': (32, 16)}


def load_data():
    with DATA.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FEATURES + ['Outcome']:
            raise ValueError('Unexpected dataset columns/order')
        rows = list(reader)
    raw = np.array([[float(row[k]) for k in FEATURES + ['Outcome']] for row in rows])
    if raw.shape != (768, 9) or not np.isfinite(raw).all():
        raise ValueError('Expected 768 finite records and 9 columns')
    if set(raw[:, -1]) != {0., 1.} or (raw[:, :-1] < 0).any():
        raise ValueError('Invalid feature values or outcome')
    x, y = raw[:, :-1].copy(), raw[:, -1].astype(int)
    zero_counts = {FEATURES[j]: int((x[:, j] == 0).sum()) for j in ZERO_MISSING}
    for j in ZERO_MISSING:
        x[x[:, j] == 0, j] = np.nan
    return raw, x, y, zero_counts


def scores(model, x, y):
    probability = model.predict_proba(x)[:, 1]
    prediction = (probability >= .5).astype(int)
    cm = confusion_matrix(y, prediction, labels=[0, 1])
    tn, fp, fn, tp = map(int, cm.ravel())
    return {
        'accuracy': float(accuracy_score(y, prediction)),
        'precision': float(precision_score(y, prediction, zero_division=0)),
        'recall': float(recall_score(y, prediction, zero_division=0)),
        'f1': float(f1_score(y, prediction, zero_division=0)),
        'specificity': tn / (tn + fp),
        'roc_auc': float(roc_auc_score(y, probability)),
        'log_loss': float(log_loss(y, probability, labels=[0, 1])),
        'confusion_matrix': cm.tolist(), 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }


def fit_network(hidden, xt, yt, xv, yv):
    # partial_fit continues Adam's optimizer state rather than restarting it.
    model = MLPClassifier(hidden_layer_sizes=hidden, activation='relu', solver='adam',
                          alpha=.0001, batch_size=32, learning_rate_init=.001,
                          shuffle=False, early_stopping=False, random_state=SEED)
    rng = np.random.RandomState(SEED)
    best_loss, significant_loss, stale = np.inf, np.inf, 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        order = rng.permutation(len(yt))
        model.partial_fit(xt[order], yt[order], classes=np.array([0, 1]))
        pt, pv = model.predict_proba(xt)[:, 1], model.predict_proba(xv)[:, 1]
        tl, vl = log_loss(yt, pt, labels=[0, 1]), log_loss(yv, pv, labels=[0, 1])
        history.append([epoch, tl, vl, accuracy_score(yt, pt >= .5), accuracy_score(yv, pv >= .5)])
        if vl < best_loss:
            best_loss, best_epoch, checkpoint = vl, epoch, deepcopy(model)
        if vl < significant_loss - MIN_DELTA:
            significant_loss, stale = vl, 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    return checkpoint, best_epoch, history


def write_csv(name, header, rows):
    with (OUT / name).open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def plot_results(histories, metrics, zeros):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    colors = ['#156b87', '#cc6735']
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7))
    for ax, (name, h) in zip(axes, histories.items()):
        h = np.asarray(h)
        ax.plot(h[:, 0], h[:, 1], label='Training', color=colors[0])
        ax.plot(h[:, 0], h[:, 2], label='Validation', color=colors[1])
        ax.axvline(metrics['models'][name]['best_epoch'], color='gray', ls=':', label='Best epoch')
        ax.set(title=name, xlabel='Epoch', ylabel='Binary cross entropy')
        ax.legend(fontsize=9); ax.grid(alpha=.18)
    fig.tight_layout(); fig.savefig(OUT / 'learning_curves.png', dpi=190); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, (name, item) in zip(axes, metrics['models'].items()):
        cm = np.array(item['test']['confusion_matrix'])
        ax.imshow(cm, cmap='Blues', vmin=0, vmax=100)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=19,
                        color='white' if cm[i, j] > 50 else '#172936')
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['0', '1'], yticklabels=['0', '1'],
               xlabel='Predicted label', ylabel='True label', title=name)
    fig.tight_layout(); fig.savefig(OUT / 'confusion_matrices.png', dpi=190); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    names = list(zeros)
    bars = ax.bar(names, list(zeros.values()), color=colors[0])
    ax.bar_label(bars, padding=3)
    ax.set(ylabel='Zero values treated as missing', ylim=(0, 430))
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / 'missing_values.png', dpi=190); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    labels = ['Accuracy', 'Precision', 'Recall', 'F1']
    x = np.arange(len(labels))
    for k, (name, item) in enumerate(metrics['models'].items()):
        vals = [item['test'][key] for key in ['accuracy', 'precision', 'recall', 'f1']]
        bars = ax.bar(x + (k - .5) * .34, vals, width=.34, label=name, color=colors[k])
        ax.bar_label(bars, labels=[f'{v:.3f}' for v in vals], padding=3, fontsize=9)
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 1.08), ylabel='Test score')
    ax.legend(loc='lower left'); ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / 'test_comparison.png', dpi=190); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw, x, y, zeros = load_data()
    indices = np.arange(len(y))
    train_val, test = train_test_split(indices, test_size=.2, stratify=y, random_state=SEED)
    train, val = train_test_split(train_val, test_size=.25, stratify=y[train_val], random_state=SEED)
    assert len(set(train) | set(val) | set(test)) == len(y)
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    # Fit both preprocessing steps ONLY on training data.
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    xt = scaler.fit_transform(imputer.fit_transform(x[train]))
    xv = scaler.transform(imputer.transform(x[val]))
    xe = scaler.transform(imputer.transform(x[test]))
    np.savez_compressed(OUT / 'split_indices.npz', train=train, validation=val, test=test)
    write_csv('splits.csv', ['row_index', 'split', 'outcome'],
              ([int(i), label, int(y[i])] for label, ids in [('train', train), ('validation', val), ('test', test)] for i in ids))
    # Outliers are inspected using training-set IQR fences, not removed.
    q1, q3 = np.nanpercentile(x[train], [25, 75], axis=0)
    iqr = q3 - q1
    outliers = np.nansum((x[train] < q1 - 1.5 * iqr) | (x[train] > q3 + 1.5 * iqr), axis=0)
    metrics = {
        'seed': SEED, 'dataset_sha256': hashlib.sha256(DATA.read_bytes()).hexdigest(),
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'scipy': scipy.__version__, 'scikit-learn': sklearn.__version__,
                        'matplotlib': matplotlib.__version__, 'joblib': joblib.__version__},
        'dataset': {'rows': len(y), 'features': FEATURES, 'class_counts': np.bincount(y).tolist(),
                    'duplicate_rows': int(len(raw) - len(np.unique(raw, axis=0))),
                    'literal_missing_values': int(np.isnan(raw).sum()), 'zero_as_missing': zeros,
                    'training_iqr_outliers': dict(zip(FEATURES, map(int, outliers)))},
        'split': {name: {'n': len(ids), 'classes': np.bincount(y[ids]).tolist()}
                  for name, ids in [('train', train), ('validation', val), ('test', test)]},
        'preprocessing': {'imputation_medians': imputer.statistics_.tolist(),
                          'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist()},
        'training': {'max_epochs': MAX_EPOCHS, 'patience': PATIENCE, 'min_delta': MIN_DELTA,
                     'batch_size': 32, 'learning_rate': .001, 'alpha': .0001,
                     'threshold': .5, 'selection': 'minimum validation log loss'},
        'models': {}
    }
    models, histories = {}, {}
    with threadpool_limits(limits=1):
        for name, hidden in ARCHITECTURES.items():
            model, epoch, history = fit_network(hidden, xt, y[train], xv, y[val])
            assert model.out_activation_ == 'logistic'
            models[name], histories[name] = model, history
            metrics['models'][name] = {
                'hidden_layers': list(hidden), 'parameters': int(sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)),
                'best_epoch': epoch, 'epochs_run': len(history),
                'train': scores(model, xt, y[train]), 'validation': scores(model, xv, y[val])}
            write_csv(name + '_history.csv', ['epoch', 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'], history)
        # Lock the architecture selection before reading test performance.
        metrics['selected_model'] = min(models, key=lambda name: metrics['models'][name]['validation']['log_loss'])
        for name, model in models.items():
            metrics['models'][name]['test'] = scores(model, xe, y[test])
            pipeline = Pipeline([('imputer', imputer), ('scaler', scaler), ('model', model)])
            joblib.dump({'pipeline': pipeline, 'features': FEATURES, 'zero_missing_indices': ZERO_MISSING,
                         'threshold': .5, 'name': name}, OUT / (name + '.joblib'))
            probability = model.predict_proba(xe)[:, 1]
            write_csv(name + '_test_predictions.csv', ['row_index', 'true', 'predicted', 'probability'],
                      zip(test, y[test], (probability >= .5).astype(int), probability))
    metrics['majority_baseline_test_accuracy'] = float(np.mean(y[test] == np.argmax(np.bincount(y[train]))))
    (OUT / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    keys = ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'roc_auc', 'log_loss']
    write_csv('comparison.csv', ['model', 'best_epoch', 'validation_log_loss'] + keys,
              ([name, m['best_epoch'], m['validation']['log_loss']] + [m['test'][k] for k in keys] for name, m in metrics['models'].items()))
    plot_results(histories, metrics, zeros)
    print(json.dumps({'selected_model': metrics['selected_model'], 'split': metrics['split'],
                      'models': metrics['models']}, indent=2))
    return metrics


if __name__ == '__main__':
    main()

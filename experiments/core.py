"""Numerical kernels reused from the archived real-data experiment.

The selected function bodies are unchanged. CLI and I/O are separate.
"""
from pathlib import Path
import datetime
import hashlib
import json
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score

REPS = [('elapsed', 128), ('index', 128), ('elapsed', 64), ('elapsed', 256)]
CONFIGS = [('elapsed', 128, 'l1'), ('elapsed', 128, 'l2'),
           ('index', 128, 'l1'), ('index', 128, 'l2'),
           ('elapsed', 64, 'l1'), ('elapsed', 256, 'l1')]
POOLS = ['mean', 'rms', 'max', 'tail_mean', 'tail_rms']
ALPHAS = [.01, .05, .1]

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


def interpolate(raw, elapsed_seconds, length, parameterization):
    raw = np.asarray(raw, dtype=float)
    elapsed_seconds = np.asarray(elapsed_seconds, dtype=float)
    assert raw.ndim == 2 and raw.shape[1] == 12
    assert len(raw) == len(elapsed_seconds) > 1
    assert np.isfinite(raw).all() and np.isfinite(elapsed_seconds).all()
    assert np.all(np.diff(elapsed_seconds) > 0), 'nonincreasing elapsed seconds'
    assert length in (64, 128, 256)
    if parameterization == 'elapsed':
        t = elapsed_seconds
    elif parameterization == 'index':
        t = np.arange(len(raw), dtype=float)
    else:
        raise ValueError(parameterization)
    grid = np.linspace(t[0], t[-1], length)
    return np.column_stack([np.interp(grid, t, raw[:, j]) for j in range(12)])


def threshold(values, alpha):
    values = np.asarray(values)
    assert len(values) > 0 and np.isfinite(values).all()
    k = int(np.ceil((len(values) + 1) * (1 - alpha)))
    return float(np.sort(values)[k - 1]) if k <= len(values) else np.inf


def pool(local):
    k = int(np.ceil(.05 * local.shape[1]))
    tail = np.partition(local, -k, axis=1)[:, -k:]
    return {'mean': local.mean(1), 'rms': np.sqrt(np.mean(local ** 2, 1)),
            'max': local.max(1), 'tail_mean': tail.mean(1),
            'tail_rms': np.sqrt(np.mean(tail ** 2, 1))}


def fit_normalization(x, fit):
    mean = x[fit].mean((0, 1))
    std = x[fit].std((0, 1))
    std[std < 1e-12] = 1
    return mean, std


def nearest(z, fit, metric):
    train = z[fit]
    distance = cdist(z.reshape(len(z), -1), train.reshape(len(train), -1),
                     metric={'l1': 'cityblock', 'l2': 'euclidean'}[metric])
    # Stable np.argmin first index preserves frozen fitting order under ties.
    refs = fit[distance.argmin(1)]
    local = np.abs(z - z[refs]).mean(2)
    return pool(local), local, refs


def simple_features(x, fit):
    speed = np.abs(x[:, :, 6:]).mean((1, 2))
    effort = np.abs(x[:, :, :6]).mean((1, 2))
    point_rms = np.sqrt(np.mean(x[:, :, 6:] ** 2, axis=2))
    quiet_threshold = float(np.quantile(point_rms[fit].ravel(), .1, method='linear'))
    quiet = np.mean(point_rms <= quiet_threshold, axis=1)
    speed_median = float(np.median(speed[fit]))
    effort_median = float(np.median(effort[fit]))
    speed_mad_raw = float(np.median(np.abs(speed[fit] - speed_median)))
    effort_mad_raw = float(np.median(np.abs(effort[fit] - effort_median)))
    speed_mad = speed_mad_raw if speed_mad_raw >= 1e-12 else 1.
    effort_mad = effort_mad_raw if effort_mad_raw >= 1e-12 else 1.
    scores = {'motion_low_speed': -speed,
              'motion_speed_mad': np.abs(speed - speed_median) / speed_mad,
              'effort_mad': np.abs(effort - effort_median) / effort_mad,
              'motion_quiet_fraction': quiet}
    fit_state = {'quiet_point_speed_rms_quantile': quiet_threshold,
                 'quantile_method': 'linear', 'speed_median': speed_median,
                 'effort_median': effort_median, 'speed_mad_raw': speed_mad_raw,
                 'effort_mad_raw': effort_mad_raw, 'speed_mad_used': speed_mad,
                 'effort_mad_used': effort_mad, 'mad_consistency_factor': 1.0,
                 'fit_n': int(len(fit)), 'fit_phase_points': int(len(fit) * 128)}
    return scores, fit_state, {'speed_mean': speed, 'effort_mean': effort,
                              'quiet_fraction': quiet}


def metrics(scores, anomaly, category, predictions):
    neg = anomaly == 0
    pos = anomaly == 1
    cats = np.unique(category[pos])
    by = {}
    for k in cats:
        hit = pos & (category == k)
        by[str(int(k))] = {'n': int(hit.sum()), 'detected': int(predictions[hit].sum()),
                           'tpr': float(predictions[hit].mean())}
    return {'n_normal': int(neg.sum()), 'false_alarms': int(predictions[neg].sum()),
            'fpr': float(predictions[neg].mean()) if neg.any() else None,
            'n_fault': int(pos.sum()), 'detected': int(predictions[pos].sum()),
            'tpr': float(predictions[pos].mean()) if pos.any() else None,
            'macro_tpr': float(np.mean([v['tpr'] for v in by.values()])) if len(cats) else None,
            'auroc': float(roc_auc_score(anomaly, scores)) if pos.any() and neg.any() else None,
            'macro_auroc': float(np.mean([
                roc_auc_score(anomaly[neg | (pos & (category == k))],
                              scores[neg | (pos & (category == k))])
                for k in cats])) if len(cats) and neg.any() else None,
            'fault_categories': by}

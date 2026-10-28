"""Export paper tables from packaged scores and archived model metrics.

All primary metrics are recomputed from archived continuous scores and checked
against original model metrics. Bootstrap intervals are archival summaries,
not new model fits or new bootstrap runs.
"""
from pathlib import Path
import csv
import hashlib
import json
import platform

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'tables'
METHODS = [
    ('nn_elapsed128_l1_mean', 'Mean'),
    ('nn_elapsed128_l1_rms', 'All RMS'),
    ('nn_elapsed128_l1_max', 'Max'),
    ('nn_elapsed128_l1_tail_mean', 'Tail mean'),
    ('nn_elapsed128_l1_tail_rms', 'Tail RMS'),
    ('motion_low_speed', 'Low speed'),
    ('motion_speed_mad', 'Speed deviation'),
    ('effort_mad', 'Effort deviation'),
    ('motion_quiet_fraction', 'Quiet fraction'),
    ('pca', 'PCA'),
    ('iforest', 'Isolation forest'),
]
CONFIGS = ['elapsed128_l1', 'elapsed128_l2', 'index128_l1',
           'index128_l2', 'elapsed64_l1', 'elapsed256_l1']
USED = set()


def load_json(relative):
    USED.add(relative)
    return json.loads((ROOT / relative).read_text())


def cohort(dataset):
    if dataset == 'factory':
        prep = 'inputs/new_external/prepared/'
        rows = load_json(prep + 'rows.json')
        split = load_json(prep + 'split.json')
        relative = 'inputs/new_external/results/factory_pickplace_v1/scores.npz'
    else:
        prep = 'inputs/existing_probe/prepared/'
        rows = load_json(prep + dataset + '_rows.json')
        split = load_json(prep + dataset + '_split.json')
        relative = 'inputs/existing_probe/results/existing_probe_v1/' + dataset + '_scores.npz'
    USED.add(relative)
    with np.load(ROOT / relative, allow_pickle=False) as data:
        scores = {key: data[key] for key in data.files}
    if dataset != 'factory':
        provenance = load_json('inputs/original/BASELINE_SCORE_PROVENANCE.json')['datasets'][dataset]
        baseline = f'inputs/original/{dataset}_baseline_scores.npz'
        for path, digest_key in [(baseline, 'output_sha256'),
                                 (prep + dataset + '_rows.json', 'aligned_rows_sha256'),
                                 (prep + dataset + '_split.json', 'aligned_split_sha256')]:
            USED.add(path)
            if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != provenance[digest_key]:
                raise ValueError('Baseline provenance checksum mismatch: ' + path)
        with np.load(ROOT / baseline, allow_pickle=False) as data:
            if set(data.files) != {'pca', 'iforest'}:
                raise ValueError('Expected exactly PCA and Isolation Forest baseline vectors')
            for key in data.files:
                if key in scores:
                    raise ValueError('Duplicate score source: ' + key)
                scores[key] = data[key]
    for key, score in scores.items():
        if score.shape != (len(rows),) or not np.isfinite(score).all():
            raise ValueError(f'Invalid score shape or nonfinite values: {dataset}/{key}')
    return rows, split, scores


def score_metrics(rows, split, score):
    y = np.array([row['anomaly'] for row in rows], dtype=int)
    cal, test = np.asarray(split['cal']), np.asarray(split['test'])
    if set(cal) & set(test) or y[cal].any():
        raise ValueError('Calibration/test split or nominal calibration violation')
    rank = int(np.ceil((len(cal) + 1) * .95))
    threshold = np.sort(score[cal])[rank - 1] if rank <= len(cal) else np.inf
    yy, ss = y[test], score[test]
    prediction = ss > threshold
    order = np.argsort(-ss, kind='stable')
    ends = np.r_[np.flatnonzero(np.diff(ss[order])), len(ss) - 1]
    tp, fp = np.cumsum(yy[order])[ends], np.cumsum(1 - yy[order])[ends]
    n_fault, n_normal = int(yy.sum()), int((1 - yy).sum())
    if not n_fault or not n_normal:
        raise ValueError('Both test classes are required for AUROC')
    detected = int(prediction[yy == 1].sum())
    false_alarms = int(prediction[yy == 0].sum())
    result = dict(n_normal=n_normal, false_alarms=false_alarms,
                  n_fault=n_fault, detected=detected,
                  auroc=float(np.trapezoid(np.r_[0, tp / n_fault],
                                          np.r_[0, fp / n_normal])),
                  threshold=float(threshold), fpr=false_alarms / n_normal,
                  tpr=detected / n_fault)
    categories = {}
    for category in sorted({str(rows[i]['category']) for i in test if y[i]}):
        chosen = np.array([str(rows[i]['category']) == category and y[i] == 1 for i in test])
        categories[category] = dict(n=int(chosen.sum()), detected=int(prediction[chosen].sum()))
    result['fault_categories'] = categories
    return result


def select(records, dataset, method):
    found = [r for r in records if r['dataset'] == dataset and r['cohort'] == 'test'
             and r['seed'] == 2027 and r['alpha'] == .05 and r['method'] == method]
    if len(found) != 1:
        raise ValueError(f'Expected one archived metric: {dataset}/{method}; found {len(found)}')
    return found[0]


def compare_metrics(actual, archived):
    for key in ['n_normal', 'false_alarms', 'n_fault', 'detected']:
        if actual[key] != archived[key]:
            raise ValueError(f'Archived {key} mismatch: {actual[key]} != {archived[key]}')
    if not np.isclose(actual['auroc'], archived['auroc'], rtol=0, atol=1e-12):
        raise ValueError('Archived AUROC mismatch')
    if not np.isclose(actual['threshold'], archived['threshold'], rtol=0, atol=1e-12):
        raise ValueError('Archived threshold mismatch')
    if set(actual['fault_categories']) != set(archived['fault_categories']):
        raise ValueError('Archived fault categories mismatch')
    for category, result in actual['fault_categories'].items():
        for key in ['n', 'detected']:
            if result[key] != archived['fault_categories'][category][key]:
                raise ValueError('Archived category counts mismatch')


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    original = load_json('inputs/original/metrics.json')
    probe = load_json('results/archived/existing_probe_metrics.json')
    factory = load_json('results/archived/factory_metrics.json')
    records, sensitivity, by_key = [], [], {}
    for dataset in ['voraus', 'ur3', 'factory']:
        rows, split, scores = cohort(dataset)
        archived_dataset = 'factory_pickplace' if dataset == 'factory' else dataset
        for method, label in METHODS:
            archive = factory if dataset == 'factory' else original if method in ['pca', 'iforest'] else probe
            recorded = select(archive, archived_dataset, method)
            if method not in scores:
                raise ValueError('Missing score vector: ' + method)
            result = score_metrics(rows, split, scores[method])
            compare_metrics(result, recorded)
            by_key[dataset, method] = result
            records.append(dict(dataset=dataset, method=method, label=label,
                                **{key: result[key] for key in ['n_normal', 'false_alarms', 'n_fault',
                                                               'detected', 'auroc', 'threshold']},
                                source='recomputed from frozen scores'))
        for config in CONFIGS:
            mean = score_metrics(rows, split, scores[f'nn_{config}_mean'])
            tail = score_metrics(rows, split, scores[f'nn_{config}_tail_rms'])
            sensitivity.append(dict(dataset=archived_dataset, config=config, mean_fa=mean['false_alarms'],
                                    tail_fa=tail['false_alarms'], mean_detected=mean['detected'],
                                    tail_detected=tail['detected'], mean_auroc=mean['auroc'],
                                    tail_auroc=tail['auroc'], tpr_delta_pp=100 * (tail['tpr'] - mean['tpr']),
                                    fpr_delta_pp=100 * (tail['fpr'] - mean['fpr'])))
    write_csv('main_table.csv', records)
    lines = []
    for method, label in METHODS:
        cells = [label]
        for dataset in ['voraus', 'ur3', 'factory']:
            row = by_key[dataset, method]
            cells.extend([str(row['false_alarms']), str(row['detected']), f"{row['auroc']:.3f}"])
        lines.append(' & '.join(cells) + r' \\')
    (OUT / 'main_rows.tex').write_text('\n'.join(lines) + '\n')
    categories, category_lines = [], []
    for method, label in METHODS[:5] + METHODS[8:]:
        row = by_key['ur3', method]
        cells = [label]
        for category, name in [('1', 'Protective stop only'), ('2', 'Grip loss only'), ('3', 'Both')]:
            item = row['fault_categories'][category]
            cells.append(f"{item['detected']}/{item['n']}")
            categories.append(dict(method=label, category=name, support=item['n'], detected=item['detected'],
                                   recall=item['detected'] / item['n'], false_alarms=row['false_alarms'],
                                   normal_support=row['n_normal']))
        cells.append(f"{row['false_alarms']}/{row['n_normal']}")
        category_lines.append(' & '.join(cells) + r' \\')
    write_csv('ur3_category_table.csv', categories)
    (OUT / 'category_table.tex').write_text(
        '\\begin{tabular}{lrrrr}\n\\hline\nMethod & PS & GL & Both & FA\\\\\n\\hline\n'
        + '\n'.join(category_lines) + '\n\\hline\n\\end{tabular}\n')
    write_csv('sensitivity.csv', sensitivity)
    uncertainty = load_json('results/archived/factory_uncertainty.json')
    block_rows = []
    for row in uncertainty['bootstrap']:
        lower, upper = row['fpr_tpr_percentiles']
        block_rows.append(dict(block_length=row['block_length'], draws=row['draws'], valid=row['valid'],
                               fpr_delta_pp=100 * uncertainty['point']['fpr_delta'],
                               fpr_lower_pp=100 * lower[0], fpr_upper_pp=100 * upper[0],
                               tpr_delta_pp=100 * uncertainty['point']['tpr_delta'],
                               tpr_lower_pp=100 * lower[1], tpr_upper_pp=100 * upper[1],
                               source='archived paired circular-block bootstrap percentiles'))
    write_csv('factory_block_intervals.csv', block_rows)
    output_names = ['main_table.csv', 'main_rows.tex', 'ur3_category_table.csv', 'category_table.tex',
                    'sensitivity.csv', 'factory_block_intervals.csv']
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(python=platform.python_version(), numpy=np.__version__,
                    inputs={name: digest(ROOT / name) for name in sorted(USED)},
                    outputs={name: digest(OUT / name) for name in output_names},
                    model_fitting=False, bootstrap_rerun=False,
                    primary_rows=len(records), sensitivity_rows=len(sensitivity))
    (OUT / 'export_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Exported 33 primary metric rows, 24 category rows, 18 sensitivity rows and 3 archived block intervals to tables/.')


if __name__ == '__main__':
    main()

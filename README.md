# Testing Temporal Tail Pooling for Robot Anomaly Detection

Research code and archived numerical evidence for comparing temporal aggregation in robot anomaly detection. The study fixes a nominal reference trajectory, computes local sensor discrepancies, and compares mean, all-point RMS, maximum, tail mean, and tail RMS aggregation on voraus-AD, UR3 CobotOps, and FactoryNet.

## Reproduce the paper diagnostics

Python 3.12 is recommended. The included score files and labels are sufficient for the diagnostic analysis; raw sensor data are acquired separately.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py verify
python reproduce.py analysis
python reproduce.py figures
```

`analysis` regenerates six CSVs and requires byte-for-byte agreement with the archived references. `figures` produces the low-FPR ROC frontiers and nominal-score ECDFs. It requires Arial or DejaVu Sans; otherwise set `DIAGNOSTIC_FONT` to a readable TrueType font file. Font differences affect rendered PDFs, not numerical results.

## Repository contents

- `experiments/`: source acquisition, preprocessing and model-scoring entry points.
- `analysis/`: frozen-score analysis and vector figure generation.
- `inputs/`: archived scores, labels, split indices and original category metrics.
- `results/reference/`: the six reference diagnostic CSVs.
- `results/archived/`: original model metrics, FactoryNet uncertainty/cost and six-configuration sensitivity results.
- `tests/`: small implementation checks using explicitly synthetic inputs.
- `DATA.md`: source versions, preprocessing and full-experiment commands.
- `ATTRIBUTION.md`: dataset credits and applicable terms.

## Experimental interpretation

Each record uses six effort and six speed channels. References are selected using whole-trajectory distance, and all five pools share the selected reference. The tail contains the largest `ceil(0.05 * L)` local discrepancies. The primary representation has 128 elapsed-time points. The source code also covers interpolation and reference-distance sensitivity, simple motion/effort rules, PCA and Isolation Forest.

Nominal calibration sets determine method-specific thresholds. Full AUROC, low-FPR ranking, and category coverage answer different questions. The additional ROC and distribution diagnostics were specified after previous test inspection; their protocol is in `analysis/POSTHOC_PROTOCOL.json`. Test-derived ROC thresholds are retrospective summaries and do not replace the original calibration thresholds. UR3 has 12 normal test records, so its observed FPR grid has increments of 1/12.

## Results reproduced by the quick analysis

At the primary nominal 5% calibration setting, FactoryNet mean pooling detects 239/975 faults with 80/663 nominal alarms; tail RMS detects 38/975 with 34/663 alarms. On UR3, retrospective zero-normal-alarm decisions detect 0/31 events with mean, 5/31 with tail RMS, and 10/31 with maximum. The category table retains grip-loss events missed by mean but detected by quiet fraction.

## Verification of this release

The six numerical diagnostic CSVs reproduce byte for byte. All four voraus-AD and UR3 preprocessing arrays reproduce the archived arrays exactly. A real-data UR3 model replay reproduces all 34 pooling/rule score vectors exactly. PCA and Isolation Forest scores differ slightly under the pinned portable environment; their primary false-alarm counts, detections and AUROC match the paper. See `experiments/PORTABILITY_CHECK.json` for numerical differences and the checks performed. FactoryNet candidate IDs, order, partitions and labels were compared with the archived cohort. A fresh full download and all-three-source model retraining were not repeated for this release.

## Data and reuse

Public data sources retain their own licenses. See `ATTRIBUTION.md` before reusing derived scores or labels. Raw sensor datasets, third-party papers, and the conference template are not bundled. No separate open-source license is granted for the study code in this release.

The repository contains the program and numerical evidence. Paper author identities, publication metadata and citation identifiers are added when confirmed by the authors.

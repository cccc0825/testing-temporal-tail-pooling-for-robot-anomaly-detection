# Data attribution and terms

These notices apply to data-derived score, label, split and result files, separately from the study source code. No endorsement by the data creators is implied.

## voraus-AD

Jan Thieß Brockmann, Marco Rudolph, Bodo Rosenhahn and Bastian Wandt. *The voraus-AD Dataset for Anomaly Detection in Robot Applications*. IEEE Transactions on Robotics 40, 438–451 (2024). DOI: https://doi.org/10.1109/TRO.2023.3332224

Source: https://github.com/vorausrobotik/voraus-ad-dataset

Data terms: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The official repository's code has separate MIT terms; this does not change the dataset terms.

## UR3 CobotOps

Marios Tyrovolas, Khurshid Aliev, Dario Antonelli and Chrysostomos Stylios. *UR3 CobotOps*. UCI Machine Learning Repository (2024). DOI: https://doi.org/10.24432/C5J891

Source: https://archive.ics.uci.edu/dataset/963/ur3+cobotops

Data terms: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## FactoryNet

Karim Othman, Jonas Petersen, Matei Ignuta-Ciucanu, Camilla Mazzoleni, Federico Martelli, Alessandro Lombardi, Riccardo Maggioni and Philipp Petersen. *FactoryNet: A large-scale dataset toward industrial time-series foundation models* (2026), https://arxiv.org/abs/2605.09081

Source: https://huggingface.co/datasets/Forgis/FactoryNet

Pinned revision: `386d731fab807f53ac17a69cf209a2b42b491072`.

The archived repository labels the data CC BY-NC-SA 4.0; the associated paper describes MIT terms. This release follows the repository's [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) terms and retains the discrepancy rather than asserting unrestricted MIT rights.

## Transformations and file mapping

The study selects effort/speed channels, partitions and interpolates records, fits on nominal records, and exports scores, labels, split indices and derived evaluation statistics. See `DATA.md` for the protocol and exclusions.

- `inputs/existing_probe/**/voraus_*`: voraus-AD derived data, CC BY-NC-SA 4.0.
- `inputs/existing_probe/**/ur3_*`: UR3 CobotOps derived data, CC BY 4.0.
- `inputs/original/voraus_baseline_scores.npz`: voraus-AD derived PCA/Isolation Forest scores, CC BY-NC-SA 4.0.
- `inputs/original/ur3_baseline_scores.npz`: UR3 CobotOps derived PCA/Isolation Forest scores, CC BY 4.0.
- `inputs/new_external/**`: FactoryNet derived data, CC BY-NC-SA 4.0.
- `inputs/original/metrics.json` and `results/reference/*`: combined source-specific metrics. The source-specific terms remain attached to their respective portions; preserve all notices when redistributing the combined files.

The source links and terms above were documented during the study's September 2026 acquisition. The package makes no new licensing claim on the underlying datasets. Study code has no separate open-source license assigned in this release.

# Data acquisition and model reproduction

The repository contains program source, frozen scores and aggregate results. Raw telemetry and prepared sensor arrays are obtained separately under the original dataset terms. Run commands below from the repository root with Python 3.12. Output directories must be new: preparation and model execution refuse to overwrite earlier runs.

## Environment

```sh
python -m pip install -r experiments/requirements.txt
python -m tests.test_models
```

If `tests` is shadowed by another installed package, run `PYTHONPATH=. python tests/test_models.py`. The tests use deliberately synthetic fixtures to check program behavior; these fixtures do not enter the research results.

This portable replay environment is separate from the recorded historical environment. PCA/SVD and Isolation Forest results should be compared with the archived scores when using another numerical-library version; the release does not promise bitwise model equality across arbitrary versions or hardware.

## voraus-AD

Source: [official dataset repository](https://github.com/vorausrobotik/voraus-ad-dataset), [100 Hz Parquet](https://media.vorausrobotik.com/voraus-ad-dataset-100hz.parquet). Citation: Jan Thieß Brockmann, Marco Rudolph, Bodo Rosenhahn and Bastian Wandt, *The voraus-AD Dataset for Anomaly Detection in Robot Applications*, IEEE Transactions on Robotics 40, 438–451 (2024), DOI 10.1109/TRO.2023.3332224. **Dataset: CC BY-NC-SA 4.0.** The official repository's MIT code license is distinct from the data license.

The portable acquisition script reads selected Parquet columns with validated HTTP ranges. Historical source validators were size 1,115,942,833 bytes and ETag `69d8ea42-4283efb1`; no historical full-file SHA256 was computed. The program checks the strong ETag, Content-Range, byte count and every cached column hash. A changed source stops the pinned reconstruction.

```sh
python experiments/acquire_voraus.py --output-columns data/voraus-columns --manifest data/voraus-download.json --expected-etag 69d8ea42-4283efb1 --expected-content-length 1115942833
python -m experiments.prepare voraus --source data/voraus-columns --output prepared/voraus
python -m experiments.run voraus --prepared prepared/voraus --output runs/voraus-2027
```

The source contains 2,122 cycles. Nominal training setting 72 contributes 948 cycles, permuted with NumPy seed 2027 into 568 fitting, 190 calibration and 190 audit records. The official test set contains 419 normal and 755 fault records. Preparation uses six torque and six velocity channels in original precision. Each grid is interpolated directly from original frames. Run preparation again with `--seed 2028` through `--seed 2031` into distinct directories to reproduce the earlier fitting/calibration split sensitivity; use the matching `--seed` for the model's Isolation Forest. These splits reuse the same test records.

## UR3 CobotOps

Source: [UCI UR3 CobotOps](https://archive.ics.uci.edu/dataset/963/ur3+cobotops), DOI 10.24432/C5J891. Creators: Marios Tyrovolas, Khurshid Aliev, Dario Antonelli and Chrysostomos Stylios (2024). **CC BY 4.0.** The downloader verifies the archived ZIP SHA256 `233334eeb5370e594ea621796ca2d693bd4980cc2bb24fa749b1b633a4993c8d`.

```sh
python -m experiments.download ur3 --output data/ur3
python -m experiments.prepare ur3 --source data/ur3/ur3_source.zip --output prepared/ur3
python -m experiments.run ur3 --prepared prepared/ur3 --output runs/ur3-2027
```

The 7,409 rows form 249 contiguous cycle-counter blocks. Original blocks 0–98, 99–173 and 174–248 define fitting/calibration/test before exclusions. Missing selected signals or labels, gaps over 10 seconds, boundary blocks and nonnominal development blocks are logged in `all_rows.json`. Eligible counts are 63 fitting, 41 calibration and 43 test (12 normal, 31 events). The six current channels precede the six speed channels. Timestamps are parsed into elapsed seconds explicitly.

## FactoryNet

Source: [Forgis/FactoryNet](https://huggingface.co/datasets/Forgis/FactoryNet), revision `386d731fab807f53ac17a69cf209a2b42b491072`. Dataset paper: Karim Othman et al., *FactoryNet: A large-scale dataset toward industrial time-series foundation models*, arXiv:2605.09081 (2026). Repository terms recorded for this study are **CC BY-NC-SA 4.0**; the paper's conflicting MIT description is retained as a provenance discrepancy. Use the repository data terms; this release does not relicense telemetry.

The new portable download route obtains the complete 992,490,317-byte telemetry Parquet and verifies its published full SHA256, plus the pinned metadata Parquet. The historical experiment acquired a selected-column sparse transport instead; the full-download route is a release-time portability change. Do not supply the sparse carrier to the new full-file preparation command.

```sh
python -m experiments.download factory --output data/factory
python -m experiments.download factory-metadata --output data/factory
python -m experiments.prepare factory --source data/factory/factorywave_ur_consolidated.parquet --metadata data/factory/factorywave_data_episode_metadata.parquet --output prepared/factory
python -m experiments.run factory --prepared prepared/factory --output runs/factory-2027
```

Full source SHA256: `75e3dc82e50f2ec7bc1621b4355ccd213c530c6b9901e1bce6de9510cbb444ec`. Metadata SHA256: `6feab3cfd127a0cd9e1564c80159d6bd01d7309abb2ec25678fa8b4fe1f2b0e2`.

The source filter first intersects metadata with UUIDs actually present in the pinned consolidated telemetry file, then selects real pick-and-place, flow `00000000-0000-0000-0000-000000000010`, counterfactual 0. It yields 4,308 candidate episodes. Normal April 6 metadata records are ordered chronologically and divided 60/40 before quality exclusions; April 7 is the primary test, later fault-only dates form the stress cohort. Per UUID, raw timestamps are stably sorted, relative differences are multiplied by 1000, and duplicate timestamps, fewer than two frames, nonfinite channels/times and gaps over one second are excluded. First-day faults are excluded from development. Counts are 226 fitting, 150 calibration, 663 normal/975 fault primary test and 2,219 fault-only stress episodes. Metadata chronology is a database-date proxy, not independent acquisition-session identity.

## Outputs and evidence boundary

Preparation writes all four `elapsed64`, `elapsed128`, `elapsed256`, `index128` arrays, eligible `rows.json`, `split.json`, complete `all_rows.json` exclusion ledger and a completion/failure manifest with hashes. It calculates no anomaly scores. Existing prepared arrays are validated by their manifest before model replay.

The model entry reuses the archived numerical kernel bodies for normalization, reference selection, five pools, four simple rules, calibration and metrics. It evaluates all six interpolation/selector settings; additionally fits PCA (95% variance, full SVD) and Isolation Forest (200 trees, seed 2027). It writes every continuous score, prediction, category metric, model state, cost and output hash to a fresh directory. FactoryNet additionally writes the three paired circular-block sensitivity analyses (2,000 draws each).

`experiments/SOURCE_LINEAGE.json` documents reused source hashes and new interfaces. Public CLI code was adapted for relocation and removes dependence on historical internal authorization files. These changes leave historical run files intact. Synthetic CLI validation, local data-preparation comparison and full real-data model reruns are distinct verification levels; consult `experiments/PORTABILITY_CHECK.json` for what was actually executed in this release. The release checks reconstructed all four voraus/UR3 arrays and split indices exactly from local source inputs, matched all 4,308 FactoryNet cohort assignments, and reran all 36 UR3 scoring methods. Its 34 pooling/rule score arrays match exactly; PCA and Isolation Forest have small version-dependent score differences while the primary table metrics agree. A subsequent fresh-clone verification downloaded the UR3 ZIP from UCI, checked its pinned SHA256, and executed the documented preparation and all 36 scoring methods in a new Python 3.12 virtual environment. The 34 pooling/rule vectors and 102 corresponding historical metric records matched exactly. Full raw-data download and model execution for voraus-AD and FactoryNet were not repeated in that check.

# MARGAD

Code for **Learning Reliable Multiscale References for Unsupervised Graph Anomaly Detection**, prepared for *IEEE Transactions on Knowledge and Data Engineering (TKDE)*.

MARGAD is an unsupervised node-level graph anomaly detector built from three complementary signals:

- **TA-DiffRef** constructs probability-calibrated multiscale references. The selected final operator retains target-return paths; target anonymization is an F3 control in the format study.
- **WaveShift** measures the discrepancy between shared and channel-specific spectral responses.
- **Global Deviation** measures how far a node representation departs from the dominant population.

> **AI-assisted development disclosure.** Repository organization, this README, and concise code documentation were prepared with assistance from OpenAI Codex. The AI-assisted material should be reviewed by the authors before public release, and responsibility for the scientific claims and released code remains with the authors. The executable model, training code, and experiment utilities have since been synchronized with the later research checkout; this statement is not a claim that behavior remained unchanged.

## Paper-to-code mapping

The implementation retains the historical internal names `alpha`, `beta`, and `gamma` so that existing commands, artifacts, and checkpoints remain compatible.

| Paper component | Historical code name | Main implementation |
|---|---|---|
| TA-DiffRef | `alpha` | `AdaptiveWaveletAffinity` in `model.py` |
| Global Deviation | `beta` | `embedding_compactness` in `training_common.py` |
| WaveShift | `gamma` | `MultiFilterGammaWavelet` and WaveShift helpers in `model.py` / `training_common.py` |
| Joint training and score fusion | `alpha + beta + gamma` | `full_graph.py` and `large_graph.py` |

In the main model, `--alpha` weights TA-DiffRef, `--beta` corresponds to the Global Deviation coefficient, and `--gamma` corresponds to the WaveShift coefficient. The paper fixes the TA-DiffRef coefficient to one. `--alpha_mode learned_no_target_anonymization` is the selected final operator: it combines calibrated `P H` and `P² H` without deleting the target-return term. The historical `--alpha_mode full` implements the target-anonymized F3 control; its name is retained for existing checkpoints.

## Repository structure

### Core runtime

| File | Purpose |
|---|---|
| `run.py` | Main training and evaluation entry point |
| `args.py` | Command-line arguments and validation |
| `dataset_config.py` | Dataset routing, fallback defaults, and representation policies |
| `Dataloader.py` | MATLAB, Elliptic CSV, and DGL graph loading |
| `model.py` | Encoder, TA-DiffRef, and WaveShift modules |
| `training_common.py` | Global Deviation, score fusion, metrics, seeding, and reporting |
| `full_graph.py` | Full-graph training for Facebook, Reddit, Amazon, YelpChi, Elliptic, and T-Finance |
| `large_graph.py` | Neighbor-sampled training and streaming inference for T-Social |
| `utils.py` | Sparse conversion, graph Laplacian, and negative sampling utilities |
| `ablation_diagnostics.py` | Compact JSON diagnostics shared by the training paths |

### Experiment utilities

| File | Purpose |
|---|---|
| `ablation_config.py` | Declarative loss and mechanism-ablation matrix |
| `ablation_runner.py` | Resumable per-seed ablation launcher |
| `ablation_aggregate.py` | CSV/JSON aggregation of completed ablations |
| `beta_gamma_weight_search.py` | Staged Global Deviation/WaveShift weight search |
| `gamma_only_uncentered_search.py` | Uncentered WaveShift weight search from reused beta candidates |
| `hidden_dim_search.py` | Hidden-dimension sensitivity search |
| `epoch_budget_search.py` | Label-free training-budget search |
| `final_10run_efficiency.py` | Final ten-run evaluation and efficiency collection |
| `export_tsocial_scores.py` | T-Social score export with checkpoint-policy validation |
| `export_tsocial_uncentered_scores.py` | Fixed uncentered T-Social score export |
| `reuse_completed_non_tsocial_results.py` | Validation and reuse of resumable non-T-Social search artifacts |
| `ta_diffref_ablation_config.py`, `run_ta_diffref_ablation.py`, `aggregate_ta_diffref_ablation.py` | Historical A0--A6 mechanism study; not the final F0--F5 format study |
| `ta_diffref_format_config.py`, `run_ta_diffref_format.py`, `aggregate_ta_diffref_format.py` | F0--F5 operator-format definitions, execution, and aggregation |
| `tests/` | Focused argument, loss, and TA-DiffRef operator tests |

The manuscript source and PDFs are distributed separately from this code repository.

## Supported datasets

The TKDE manuscript reports results on six datasets: **Facebook, Reddit, YelpChi, T-Finance, Elliptic, and T-Social**. The runtime also retains support for **Amazon** as an additional dataset.

By default, `run.py` reads data from `dataset/`. The expected layout is:

```text
dataset/
├── Facebook.mat
├── Reddit.mat
├── Amazon.mat
├── YelpChi.mat
├── elliptic/
│   └── elliptic_bitcoin_dataset/
│       ├── elliptic_txs_features.csv
│       ├── elliptic_txs_classes.csv
│       └── elliptic_txs_edgelist.csv
├── tfinance/
│   └── tfinance
└── tsocial/
    └── tsocial
```

MATLAB datasets may use common field aliases for adjacency (`Network`, `A`, `homo`, or `adj`), features (`Attributes`, `X`, `features`, `feature`, or `feat`), and labels (`Label`, `gnd`, `label`, `labels`, or `y`). DGL graphs must contain node features and labels under one of the aliases accepted by `Dataloader.py`.

Datasets are not included in this repository. Download and use them according to their original licenses and terms.

## Environment

The manuscript experiments used:

- Python 3.9
- PyTorch 2.0.0 with CUDA 11.8
- DGL 1.1.2
- one NVIDIA GeForce RTX 4090 D with 24 GB memory

The code also imports NumPy, SciPy, pandas, and scikit-learn. Install PyTorch and DGL using builds that match the local CUDA toolkit, then install the remaining Python dependencies. A typical environment starts with:

```bash
conda create -n margad python=3.9 -y
conda activate margad
pip install numpy scipy pandas scikit-learn
```

Install the CUDA-compatible PyTorch 2.0.0 and DGL 1.1.2 packages using their official platform-specific instructions. A CPU environment is sufficient for argument parsing and dry runs, but the reported large-graph experiments require a CUDA-capable GPU.

## Quick start

Run commands from the repository root. `--data_dir` defaults to `dataset`, and `--device` defaults to `cuda`.

The following commands use the fixed dataset hyperparameters and the current retained-return default. They do not reproduce the earlier target-anonymized Table II row without adding `--alpha_mode full`.

```bash
# Facebook
python run.py --dataset 'Facebook' --hidden_dim 64 --lr 3e-3 --epoch 70 --alpha 1 --beta 0.15 --gamma 0.5
# Amazon
python run.py --dataset 'Amazon' --hidden_dim 64 --lr 3e-3 --epoch 70 --alpha 1 --beta 0.4 --gamma 0.15
# Reddit
python run.py --dataset 'Reddit' --hidden_dim 128 --lr 3e-3 --epoch 110 --alpha 1 --beta 0.35 --gamma 1.45
# YelpChi
python run.py --dataset 'YelpChi' --hidden_dim 64 --lr 3e-3 --epoch 65 --alpha 1 --beta 0.15 --gamma 1.05
# elliptic
python run.py --dataset elliptic --hidden_dim 64 --lr 1e-3 --epoch 70 --alpha 1 --beta 0.3 --gamma 1
# tfinance (full-graph training)
python run.py --dataset tfinance --hidden_dim 64 --lr 3e-3 --epoch 85 --alpha 1.0 --beta 1.00 --gamma 0.05
# tsocial (uncentered WaveShift, neighbor-sampled training)
python run.py --dataset tsocial --hidden_dim 64 --lr 3e-3 --epoch 10 --alpha 1.0 --beta 0.85 --gamma 0.75 --gamma_centering 0 --batch_size 51200 --eval_batch_size 51200 --batch_fanout 8 --num_workers 0 --dgl_graph_on_gpu 1
```

The WaveShift centering switch can be set explicitly:

```bash
python run.py --dataset Amazon --gamma_centering 1
python run.py --dataset tsocial --gamma_centering 0
```

To repeat a configuration with consecutive seeds, append `--runs` and optionally `--seed_offset`. For example, `--runs 10 --seed_offset 0` runs seeds 0 through 9.

### Manuscript-aligned ten-run evaluation

For the six datasets reported in the TKDE manuscript, use the final runner so that all final parameters and the WaveShift centering policy are passed explicitly:

```bash
python final_10run_efficiency.py \
  --datasets Facebook Reddit YelpChi tfinance elliptic tsocial \
  --data_dir dataset \
  --device cuda:0 \
  --results_root results/final_retained_return_10run
```

This launcher passes the retained-return operator explicitly, runs ten independent seeds per dataset, keeps each dataset in an isolated directory, records logs and configuration metadata, supports resumable execution by default, and writes aggregate CSV files. Use a new `--results_root` when changing the operator so earlier outputs are not reused. Preview the complete job without launching training:

```bash
python final_10run_efficiency.py \
  --datasets Facebook Reddit YelpChi tfinance elliptic tsocial \
  --dry_run
```

## Training and evaluation protocol

- Training is unsupervised; the three branch losses are optimized without anomaly labels.
- The selected checkpoint minimizes the weighted unsupervised loss during the second half of training.
- Labels are used to compute final AUROC and trapezoidal AUPRC.
- The optional epoch-level AUROC monitor is reporting-only and does not select the checkpoint. Use `--disable_monitor_auc` when the training log must remain fully label-free.
- Final metrics are reported as mean and population standard deviation (`ddof=0`) across runs.
- Elliptic propagates over all transactions but evaluates only transactions with known labels.
- T-Social uses two sampled layers, fanout 8, 51,200-node training/evaluation batches, and streaming population statistics in the preserved final configuration.

## Ablations and diagnostics

The ablation matrix covers branch combinations, TA-DiffRef mechanisms, and WaveShift mechanisms. Always preview a large matrix before launching GPU jobs:

```bash
python ablation_runner.py \
  --studies all \
  --datasets Facebook Reddit YelpChi tfinance elliptic tsocial \
  --seeds 0-9 \
  --device cuda:0 \
  --dry-run
```

Remove `--dry-run` and add `--resume` to execute or continue the selected matrix. Each seed is written to an isolated directory with its command, log, metrics, diagnostics, heartbeat, and completion marker. Aggregate completed runs with:

```bash
python ablation_aggregate.py --output_dir ablation_results_retained_return
```

For a single direct run, optional machine-readable artifacts can be requested with:

```bash
python run.py --dataset Facebook \
  --result_json results/facebook/result.json \
  --diagnostics_json results/facebook/diagnostics.json
```

### F0--F5 format study

This study is separate from the main training entry point. F3 calibrates both orders and removes target return; F5 calibrates both orders and retains return. The current runner trains all six formats independently with common seeds and writes to `return/ta_diffref_format_final_f0_f5_10seed`. The archived experiment used the opposite F3/F5 labels and imported its anonymous result from the earlier Table II evaluation; do not merge those historical run directories with new outputs. The current manuscript's Table IV maps the archived ten-run results to the displayed F3/F5 labels without inventing new measurements.

Preview the study without launching training:

```bash
python run_ta_diffref_format.py --datasets Facebook Reddit YelpChi tfinance elliptic tsocial --seeds 0-9 --dry_run
```

After training, run `python aggregate_ta_diffref_format.py` to summarize all six formats. For T-Social score export, pass the checkpoint's actual operator explicitly, for example `--alpha_mode learned_no_target_anonymization` for a newly trained final checkpoint. The switch does not change checkpoint weights, so selecting the wrong operator at inference changes the score calculation.

## Outputs

Direct `run.py` executions write the selected checkpoint in the current working directory:

- `best_model.pth` for one run;
- `best_model_run<N>.pth` when `--runs` is greater than one.

Launcher scripts use isolated working directories so checkpoints and logs from different datasets, seeds, and variants do not overwrite one another. Depending on the launcher, summaries include AUROC, AUPRC, checkpoint-selection metadata, training/inference time, peak allocated/reserved GPU memory, configuration values, and run status.

Large generated artifacts are not intended for version control. Keep datasets, checkpoints, result directories, and logs outside the source tree or exclude them before publishing.

## Citation

Please cite the TKDE paper after its bibliographic record becomes available. The final BibTeX entry can be added here after acceptance or public preprint release.

## License

The original source code is released under the MIT License. This license does not cover datasets, external repositories, or third-party artifacts, which remain subject to their respective terms.

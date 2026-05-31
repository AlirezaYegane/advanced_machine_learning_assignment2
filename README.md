# TempEdge-ResGNN for Bitcoin Illicit Transaction Detection

**COMP8221 Advanced Machine Learning — Assignment 2 (Project Option 1: Real-world applications of GNNs)**
Elliptic Bitcoin transaction graph · PyTorch Geometric

This repository contains my Assignment 2 project: detecting illicit Bitcoin transactions on the Elliptic graph, framed as semi-supervised node classification. I compare classical feature-based models, a feature-only neural baseline, standard GNNs, and a custom model (**TempEdge-ResGNN**), and I report the comparison honestly — including the result that a tuned Random Forest is still the strongest overall model on these handcrafted features.

The headline outcome:

| Category | Model | Key result |
|---|---|---|
| Best overall | Random Forest | Illicit-F1 **0.8058**, PR-AUC **0.7820** |
| Best neural / GNN | TempEdge-ResGNN | Illicit-F1 **0.5760 ± 0.0672**, PR-AUC **0.6233 ± 0.0516** (5 seeds) |
| Main challenge | — | Sharp temporal distribution shift after timestep 43 |

---

## Table of contents

- [Overview](#overview)
- [Project description](#project-description)
- [File structure](#file-structure)
- [Environment and dependencies](#environment-and-dependencies)
- [How to run](#how-to-run)
- [Quick vs full mode](#quick-vs-full-mode)
- [Generated results](#generated-results)
- [Figures](#figures)
- [Reproducibility notes](#reproducibility-notes)

---

## Overview

The core idea is that illicit activity on Bitcoin isn't only visible inside a single transaction — it shows up in the *flow* around it: where the money came from, where it goes next, and what else it touches. That makes a graph representation a more natural fit than treating each transaction as an isolated table row, which is the main reason this is set up as a GNN problem rather than a plain tabular one.

The notebook walks through the full pipeline end to end: loading and preprocessing the Elliptic graph, building the temporal train/validation/test split, training every model under the same protocol, and then running the evaluation, ablation, and per-timestep error analysis that the report is built on.

[Back to top](#table-of-contents)

## Project description

Each node is a Bitcoin transaction, each directed edge is a payment-flow link, and the labelled nodes are classified as **licit** or **illicit**. Unknown-label nodes are kept in the graph so they can still contribute to message passing, but they're excluded from the supervised loss and from evaluation — treating them as "licit" would just inject wrong labels.

A few design decisions shape the whole project:

- **Transductive, semi-supervised setup.** The full graph is available for message passing; loss is only computed on labelled training nodes.
- **Temporal split, not random.** Models train on earlier timesteps and are tested on later ones, which mirrors how an AML system actually runs (using the past to flag the future). Validation nodes are carved from timesteps 30–34 so threshold tuning never touches the test period.
- **Imbalance-aware metrics.** Only ~9.76% of labelled nodes are illicit, so accuracy is not the headline metric. The project reports illicit precision/recall/F1, macro-F1, micro-F1, ROC-AUC, and PR-AUC.

The model lineup is deliberately broad so the GNN is judged against strong competition, not just weak baselines: Logistic Regression and Random Forest (classical), MLP (feature-only neural), GCN / GraphSAGE / GAT / GATv2 (standard GNNs), and the custom **TempEdge-ResGNN** (edge-aware message passing, residual gating, non-parametric timestep features, and optional Jumping Knowledge aggregation).

The Version 2 executed notebook results are the source of truth for the report.

[Back to top](#table-of-contents)

## File structure

```text
.
├── COMP8221_TempEdgeResGNN_v2.ipynb                # main notebook
├── COMP8221_TempEdgeResGNN_v2_executed_for_review.ipynb   # executed copy (source of truth for results)
├── README.md
├── 2026S1 COMP8221 Assignment 2 60957107 Alireza_Yegane.pdf   # final report
├── results/
│   ├── all_model_results.csv
│   ├── main_results_summary.csv
│   ├── ablation_results.csv
│   ├── per_timestep_metrics.csv
│   └── distribution_shift_breakdown.csv
└── figures/
    ├── class_distribution.png
    ├── timestep_distribution.png
    ├── degree_distribution.png
    ├── graph_visualisation.png
    ├── tempedge_resgnn_architecture.png
    ├── model_comparison_f1.png
    ├── model_comparison_prauc.png
    ├── learning_curves.png
    ├── pr_curves.png
    ├── confusion_matrix_best_model.png
    ├── confusion_matrix_grid.png
    ├── per_timestep_f1.png
    └── ablation_illicit_f1.png
```

> The `data/` folder is downloaded automatically on first run and is **not** part of the submission.

[Back to top](#table-of-contents)

## Environment and dependencies

The full Version 2 run was executed with:

| Package | Version |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu128 |
| PyTorch Geometric | 2.7.0 |
| scikit-learn / pandas / numpy / matplotlib / networkx | latest compatible |

Install the main dependencies with:

```bash
pip install torch torch_geometric scikit-learn pandas numpy matplotlib networkx
```

A CUDA GPU is recommended for the full run. The code falls back to CPU automatically if CUDA isn't available, but CPU execution is considerably slower — fine for a quick smoke test, painful for the full five-seed run.

[Back to top](#table-of-contents)

## How to run

Open `COMP8221_TempEdgeResGNN_v2.ipynb` and run all cells top to bottom.

There's a global configuration cell near the start. The setting that matters is:

```python
CONFIG["MODE"] = "quick"   # or "full"
```

On the first run, the notebook downloads the Elliptic Bitcoin dataset into `data/Elliptic/`. That folder is created automatically and does not need to be submitted.

[Back to top](#table-of-contents)

## Quick vs full mode

| Mode | Purpose | Seeds / epochs |
|---|---|---|
| `quick` | Smoke test — confirms the notebook runs end to end without errors | Reduced |
| `full` | Report mode — the exact setting used to generate the submitted Version 2 results | Full (5 seeds) |

Use `quick` to verify the pipeline on a new machine, then switch to `full` to reproduce the numbers in the report. The metrics in the report and in this README come from the `full` run.

[Back to top](#table-of-contents)

## Generated results

The notebook writes CSV outputs to `results/`:

| File | Contents |
|---|---|
| `all_model_results.csv` | One row per model/seed run, with validation and test metrics |
| `main_results_summary.csv` | The main model-comparison table used in the report |
| `ablation_results.csv` | One row per ablation configuration and seed |
| `per_timestep_metrics.csv` | Per-timestep illicit-F1 for the best neural/GNN model |
| `distribution_shift_breakdown.csv` | Performance before and after timestep 43 |

[Back to top](#table-of-contents)

## Figures

All figures are written to `figures/` and grouped by what they show:

- **Data / task:** `class_distribution`, `timestep_distribution`, `degree_distribution`, `graph_visualisation`
- **Model:** `tempedge_resgnn_architecture`
- **Evaluation:** `model_comparison_f1`, `model_comparison_prauc`, `learning_curves`, `pr_curves`, `confusion_matrix_best_model`, `confusion_matrix_grid`
- **Analysis:** `per_timestep_f1`, `ablation_illicit_f1`

[Back to top](#table-of-contents)

## Reproducibility notes

- Python, NumPy, and PyTorch seeds are fixed.
- Neural models are run across **five seeds** in the full Version 2 setting; reported as mean ± standard deviation.
- Threshold tuning is done **only** on the validation set, then the chosen threshold is applied to the test set.
- Feature scaling (`StandardScaler`) is fitted on **training nodes only**, to avoid leaking later-period statistics into preprocessing.
- Unknown-label nodes are retained for message passing but excluded from loss and metrics.
- Small run-to-run variation is still possible: some PyTorch Geometric scatter operations are non-deterministic on GPU.

[Back to top](#table-of-contents)

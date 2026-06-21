# Trust-FedAvg / FedBench Extended

A trust-aware federated learning benchmarking framework that extends a FedBench-style
pipeline with multi-dataset evaluation, Non-IID (Dirichlet) data partitioning, and
communication-overhead instrumentation.

**Submitted by:** CUJ Interns — Amrita Group, Central University of Jharkhand
Research Internship, National Institute of Technology Jamshedpur

See [`paper/`](paper/) for the full IEEE-format report.

## What this does

Trust-FedAvg reweights each client's contribution to the global model using a dynamic
**trust score** (combining instantaneous loss-based quality with historical training
consistency), in addition to the standard dataset-size weighting used by FedAvg. The
framework benchmarks Trust-FedAvg against standard FedAvg across:

- **3 datasets**: MNIST, FashionMNIST, CIFAR-10
- **2 data distributions**: IID and Non-IID (Dirichlet, α = 0.5)
- **2 aggregation strategies**: FedAvg, Trust-FedAvg

→ 12 total experimental configurations, 5 simulated clients, 20 communication rounds each.

It also tracks per-round **communication overhead** (estimated from model parameter
count) and **live system resource usage** (CPU/RAM/network via `psutil`).

## Project structure

```
TrustFedAvg-FedBench/
├── main.py                   # single entry point — runs all 12 experiments
├── requirements.txt
├── config/
│   └── config.py             # all hyperparameters (clients, rounds, lr, α, τ, ...)
├── models/
│   └── cnn_models.py         # CNNGrayscale (MNIST/FashionMNIST), CNNCIFAR
├── data/
│   ├── datasets.py           # dataset loading
│   └── partition.py          # IID + Dirichlet Non-IID client partitioning
├── client/
│   └── fed_client.py         # FedClient: local SGD training
├── server/
│   ├── trust_score.py        # TrustScoreCalculator
│   ├── aggregation.py        # fedavg_aggregate, trust_fedavg_aggregate
│   ├── evaluate.py           # global model evaluation
│   └── run_experiment.py     # orchestrates one full FL run
├── comm_monitor/
│   └── network_monitor.py    # comm-overhead estimation + psutil live monitoring
├── utils/
│   ├── plotting.py           # all figure generation (Fig. 1–6 from the report)
│   └── reporting.py          # summary tables, CSV export, accuracy-drop analysis
├── results/
│   ├── plots/                # generated PNG figures
│   └── tables/                # generated CSV result tables
├── notebooks/
│   └── main_experiment.ipynb # original Colab notebook (kept for reference)
└── paper/
    └── TrustFedAvg_FedBench_Report.pdf
```

## How to run

```bash
pip install -r requirements.txt
python main.py
```

This runs all 12 configurations end-to-end, prints summary/analysis tables, saves a
results CSV to `results/tables/`, and saves all figures to `results/plots/`.

## Key results (Round 20)

| Dataset      | Algorithm     | IID Acc | Non-IID Acc | Drop   |
|--------------|--------------|---------|-------------|--------|
| MNIST        | FedAvg       | 0.9929  | 0.9896      | 0.0033 |
| MNIST        | Trust-FedAvg | 0.9937  | 0.9912      | 0.0025 |
| FashionMNIST | FedAvg       | 0.9207  | 0.8984      | 0.0223 |
| FashionMNIST | Trust-FedAvg | 0.9219  | 0.9083      | 0.0136 |
| CIFAR-10     | FedAvg       | 0.7409  | 0.7163      | 0.0246 |
| CIFAR-10     | Trust-FedAvg | 0.7304  | 0.7130      | 0.0174 |

Trust-FedAvg reduces the IID→Non-IID accuracy drop by **24%** on MNIST and **39%** on
FashionMNIST relative to FedAvg. On CIFAR-10, the trust mechanism shows a smaller
relative benefit — see the report's Discussion and Limitations sections for analysis.

## Limitations

Single-seed runs, fixed (not per-dataset-tuned) trust hyperparameters, and an
honest-but-skewed threat model only (no adversarial/Byzantine clients). Full details
in the report.

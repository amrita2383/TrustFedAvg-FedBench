"""
Main entry point: runs all 12 FedBench / Trust-FedAvg experiments
(3 datasets x 2 distributions x 2 algorithms), then generates all
result tables and plots.

Usage:
    python main.py
"""

import random

import numpy as np
import torch

from config.config import CONFIG, print_config
from server.run_experiment import run_experiment
from utils.plotting import (
    plot_accuracy_loss_curves, plot_final_accuracy_bars, plot_comm_overhead,
    plot_system_metrics, plot_trust_scores, plot_dashboard,
)
from utils.reporting import (
    print_summary_table, save_summary_csv, print_iid_noniid_analysis, print_network_summary,
)
from models.cnn_models import get_model


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print_config(CONFIG)

    # Run all 12 experiments: 3 datasets x 2 distributions x 2 algorithms
    all_results = {}
    for dataset in CONFIG["datasets"]:
        all_results[dataset] = {}
        for dist in ["IID", "NonIID"]:
            for use_trust in [False, True]:
                algo = "Trust-FedAvg" if use_trust else "FedAvg"
                key = f"{dist}_{algo}"
                all_results[dataset][key] = run_experiment(
                    dataset_name=dataset,
                    device=device,
                    distribution=dist,
                    alpha=CONFIG["dirichlet_alpha"],
                    use_trust=use_trust,
                    cfg=CONFIG,
                )

    print("\n\nAll experiments complete!")

    # Tables
    summary_rows = print_summary_table(all_results, CONFIG["datasets"])
    save_summary_csv(summary_rows)
    print_iid_noniid_analysis(all_results, CONFIG["datasets"], CONFIG["dirichlet_alpha"])
    print_network_summary(all_results, CONFIG["datasets"], get_model, device)

    # Plots
    plot_accuracy_loss_curves(all_results, CONFIG["datasets"], CONFIG["num_rounds"])
    plot_final_accuracy_bars(all_results, CONFIG["datasets"])
    plot_comm_overhead(all_results, CONFIG["datasets"], CONFIG["num_rounds"])
    plot_system_metrics(all_results, CONFIG["datasets"], CONFIG["num_rounds"])
    plot_trust_scores(all_results, CONFIG["datasets"], CONFIG["num_rounds"])
    plot_dashboard(all_results, CONFIG["datasets"], CONFIG["num_rounds"])

    return all_results


if __name__ == "__main__":
    main()

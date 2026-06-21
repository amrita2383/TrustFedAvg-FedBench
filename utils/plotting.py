"""
All figure-generation functions, kept separate from experiment logic so
each plot can be regenerated independently from saved results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

COLORS = {
    "IID_FedAvg": "#2196F3",
    "IID_Trust-FedAvg": "#4CAF50",
    "NonIID_FedAvg": "#FF5722",
    "NonIID_Trust-FedAvg": "#9C27B0",
}
LINESTYLES = {
    "IID_FedAvg": "-",
    "IID_Trust-FedAvg": "--",
    "NonIID_FedAvg": "-",
    "NonIID_Trust-FedAvg": "--",
}


def plot_accuracy_loss_curves(all_results, datasets, num_rounds, save_dir="results/plots"):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("FedBench Extended: Accuracy per Round\n(IID vs Non-IID | FedAvg vs Trust-FedAvg)",
                 fontsize=14, fontweight='bold')
    rounds = list(range(1, num_rounds + 1))

    for col, dataset in enumerate(datasets):
        for row, metric in enumerate(["accuracy", "loss"]):
            ax = axes[row][col]
            for dist in ["IID", "NonIID"]:
                for algo in ["FedAvg", "Trust-FedAvg"]:
                    key = f"{dist}_{algo}"
                    vals = all_results[dataset][key][metric]
                    ax.plot(rounds, vals, color=COLORS[key], linestyle=LINESTYLES[key],
                            linewidth=2, label=f"{dist} | {algo}", alpha=0.9)
            ax.set_title(f"{dataset} — {metric.capitalize()}", fontsize=11)
            ax.set_xlabel("Round")
            ax.set_ylabel(metric.capitalize())
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/accuracy_loss_curves.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: accuracy_loss_curves.png")


def plot_final_accuracy_bars(all_results, datasets, save_dir="results/plots"):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(datasets))
    width = 0.18

    bars_data = {
        "IID\nFedAvg": [all_results[d]["IID_FedAvg"]["accuracy"][-1] for d in datasets],
        "IID\nTrust-FedAvg": [all_results[d]["IID_Trust-FedAvg"]["accuracy"][-1] for d in datasets],
        "NonIID\nFedAvg": [all_results[d]["NonIID_FedAvg"]["accuracy"][-1] for d in datasets],
        "NonIID\nTrust-FedAvg": [all_results[d]["NonIID_Trust-FedAvg"]["accuracy"][-1] for d in datasets],
    }
    bar_colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]

    for i, (label, vals) in enumerate(bars_data.items()):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=bar_colors[i],
                       alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=7.5)

    ax.set_title("Final Accuracy Comparison — IID vs Non-IID\nAll Datasets & Algorithms",
                 fontsize=13, fontweight='bold')
    ax.set_xlabel("Dataset", fontsize=11)
    ax.set_ylabel("Accuracy (Round 20)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/iid_vs_noniid_accuracy.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: iid_vs_noniid_accuracy.png")


def plot_comm_overhead(all_results, datasets, num_rounds, save_dir="results/plots"):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Communication Overhead per Round (KB)\nIID vs Non-IID", fontsize=13, fontweight='bold')
    rounds = list(range(1, num_rounds + 1))

    for col, dataset in enumerate(datasets):
        ax = axes[col]
        for dist in ["IID", "NonIID"]:
            for algo in ["FedAvg", "Trust-FedAvg"]:
                key = f"{dist}_{algo}"
                vals = all_results[dataset][key]["comm_kb"]
                ax.plot(rounds, vals, color=COLORS[key], linestyle=LINESTYLES[key],
                        linewidth=2, label=f"{dist}|{algo}", alpha=0.9)
        ax.set_title(f"{dataset}", fontsize=11)
        ax.set_xlabel("Round")
        ax.set_ylabel("KB")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/comm_overhead.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: comm_overhead.png")


def plot_system_metrics(all_results, datasets, num_rounds, save_dir="results/plots"):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("System Metrics: CPU % and RAM (MB)\nAll Datasets", fontsize=13, fontweight='bold')
    rounds = list(range(1, num_rounds + 1))

    for col, dataset in enumerate(datasets):
        for row, metric in enumerate(["cpu_percent", "ram_mb"]):
            ax = axes[row][col]
            for dist in ["IID", "NonIID"]:
                for algo in ["FedAvg", "Trust-FedAvg"]:
                    key = f"{dist}_{algo}"
                    vals = all_results[dataset][key][metric]
                    ax.plot(rounds, vals, color=COLORS[key], linestyle=LINESTYLES[key],
                            linewidth=1.8, label=f"{dist}|{algo}", alpha=0.9)
            ylabel = "CPU (%)" if metric == "cpu_percent" else "RAM (MB)"
            ax.set_title(f"{dataset} — {ylabel}", fontsize=10)
            ax.set_xlabel("Round")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/system_metrics.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: system_metrics.png")


def plot_trust_scores(all_results, datasets, num_rounds, save_dir="results/plots"):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Average Trust Score per Round\n(Trust-FedAvg only)", fontsize=13, fontweight='bold')
    rounds = list(range(1, num_rounds + 1))

    for col, dataset in enumerate(datasets):
        ax = axes[col]
        for dist in ["IID", "NonIID"]:
            key = f"{dist}_Trust-FedAvg"
            vals = all_results[dataset][key]["trust_scores"]
            color = "#4CAF50" if dist == "IID" else "#9C27B0"
            ax.plot(rounds, vals, color=color, linewidth=2, label=f"{dist}", alpha=0.9)
        ax.set_title(f"{dataset}", fontsize=11)
        ax.set_xlabel("Round")
        ax.set_ylabel("Avg Trust Score")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.1, color='red', linestyle=':', linewidth=1, label='Threshold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/trust_scores.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: trust_scores.png")


def plot_dashboard(all_results, datasets, num_rounds, save_dir="results/plots"):
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("FedBench Extended — Complete Results Dashboard\n"
                 "MNIST | FashionMNIST | CIFAR-10 | IID vs Non-IID | FedAvg vs Trust-FedAvg",
                 fontsize=14, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    metric_labels = {"accuracy": "Accuracy", "loss": "Loss", "comm_kb": "Comm (KB)"}
    rounds = list(range(1, num_rounds + 1))

    for row, metric in enumerate(["accuracy", "loss", "comm_kb"]):
        for col, dataset in enumerate(datasets):
            ax = fig.add_subplot(gs[row, col])
            for dist in ["IID", "NonIID"]:
                for algo in ["FedAvg", "Trust-FedAvg"]:
                    key = f"{dist}_{algo}"
                    vals = all_results[dataset][key][metric]
                    ax.plot(rounds, vals, color=COLORS[key], linestyle=LINESTYLES[key],
                            linewidth=1.8, alpha=0.9, label=f"{dist}|{algo}")
            ax.set_title(f"{dataset} — {metric_labels[metric]}", fontsize=9)
            ax.set_xlabel("Round", fontsize=8)
            ax.set_ylabel(metric_labels[metric], fontsize=8)
            ax.legend(fontsize=5.5)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

    legend_elements = [
        plt.Line2D([0], [0], color="#2196F3", lw=2, label="IID | FedAvg"),
        plt.Line2D([0], [0], color="#4CAF50", lw=2, linestyle='--', label="IID | Trust-FedAvg"),
        plt.Line2D([0], [0], color="#FF5722", lw=2, label="NonIID | FedAvg"),
        plt.Line2D([0], [0], color="#9C27B0", lw=2, linestyle='--', label="NonIID | Trust-FedAvg"),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=9, bbox_to_anchor=(0.5, 0.01))

    plt.savefig(f"{save_dir}/dashboard.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: dashboard.png")

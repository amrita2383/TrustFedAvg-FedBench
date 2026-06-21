"""
Tabular and printed-analysis summaries derived from all_results.
Pure reporting code, no training/aggregation logic here.
"""

import csv

import numpy as np
import psutil


def print_summary_table(all_results, datasets):
    print("\n" + "=" * 80)
    print("SUMMARY RESULTS TABLE — Final Round")
    print("=" * 80)
    print(f"{'Dataset':<14} {'Distribution':<10} {'Algorithm':<14} "
          f"{'Accuracy':>10} {'Loss':>8} {'Comm(KB)':>10}")
    print("-" * 80)

    summary_rows = []
    for dataset in datasets:
        for dist in ["IID", "NonIID"]:
            for algo in ["FedAvg", "Trust-FedAvg"]:
                key = f"{dist}_{algo}"
                h = all_results[dataset][key]
                acc, loss, comm = h["accuracy"][-1], h["loss"][-1], sum(h["comm_kb"])
                print(f"{dataset:<14} {dist:<10} {algo:<14} {acc:>10.4f} {loss:>8.4f} {comm:>10.1f}")
                summary_rows.append({"dataset": dataset, "dist": dist, "algo": algo,
                                      "accuracy": acc, "loss": loss, "comm_kb": comm})
    print("=" * 80)
    return summary_rows


def save_summary_csv(summary_rows, path="results/tables/summary_results.csv"):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved: {path}")


def print_iid_noniid_analysis(all_results, datasets, dirichlet_alpha):
    print("\n" + "=" * 70)
    print("ANALYSIS: IID vs Non-IID Accuracy Drop")
    print("=" * 70)
    print(f"\nDirichlet alpha = {dirichlet_alpha} (lower alpha = more non-IID, more skewed)\n")
    print(f"{'Dataset':<14} {'Algorithm':<14} {'IID Acc':>9} {'NonIID Acc':>12} {'Drop':>8} {'Trust Gain':>12}")
    print("-" * 70)

    for dataset in datasets:
        for algo in ["FedAvg", "Trust-FedAvg"]:
            iid_acc = all_results[dataset][f"IID_{algo}"]["accuracy"][-1]
            noniid_acc = all_results[dataset][f"NonIID_{algo}"]["accuracy"][-1]
            drop = iid_acc - noniid_acc

            if algo == "Trust-FedAvg":
                fedavg_noniid = all_results[dataset]["NonIID_FedAvg"]["accuracy"][-1]
                trust_gain = noniid_acc - fedavg_noniid
                gain_str = f"{trust_gain:+.4f}"
            else:
                gain_str = "  baseline"

            print(f"{dataset:<14} {algo:<14} {iid_acc:>9.4f} {noniid_acc:>12.4f} "
                  f"{drop:>8.4f} {gain_str:>12}")
    print("=" * 70)


def print_network_summary(all_results, datasets, get_model_fn, device):
    print("\n" + "=" * 70)
    print("NETWORK TRAFFIC SUMMARY")
    print("=" * 70)
    print("\nModel size estimation (float32 = 4 bytes per parameter):\n")

    for dataset in datasets:
        model = get_model_fn(dataset, device)
        params = sum(p.numel() for p in model.parameters())
        size_kb = params * 4 / 1024
        print(f"  {dataset:<14}: {params:>10,} params = {size_kb:.1f} KB per model")

    print(f"\n{'Dataset':<14} {'Algorithm':<14} {'Distribution':<12} "
          f"{'Per Round KB':>14} {'Total KB':>10} {'Total MB':>10}")
    print("-" * 75)

    for dataset in datasets:
        for dist in ["IID", "NonIID"]:
            for algo in ["FedAvg", "Trust-FedAvg"]:
                key = f"{dist}_{algo}"
                per_round = all_results[dataset][key]["comm_kb"][0]
                total = sum(all_results[dataset][key]["comm_kb"])
                print(f"{dataset:<14} {algo:<14} {dist:<12} "
                      f"{per_round:>14.1f} {total:>10.1f} {total / 1024:>10.3f}")
    print("=" * 75)

    net = psutil.net_io_counters()
    print(f"\npsutil live network counter (system-wide, not per-model):")
    print(f"   Total bytes sent:     {net.bytes_sent / (1024 ** 2):.1f} MB")
    print(f"   Total bytes received: {net.bytes_recv / (1024 ** 2):.1f} MB")

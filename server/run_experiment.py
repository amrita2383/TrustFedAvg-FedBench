"""
run_experiment: orchestrates one complete federated learning run
(data loading -> partitioning -> client setup -> R communication rounds -> metrics).
"""

import time

import numpy as np
import psutil
from torch.utils.data import DataLoader

from config.config import CONFIG
from data.datasets import load_dataset
from data.partition import iid_partition, noniid_partition
from models.cnn_models import get_model
from client.fed_client import FedClient
from server.trust_score import TrustScoreCalculator
from server.aggregation import fedavg_aggregate, trust_fedavg_aggregate
from server.evaluate import evaluate
from comm_monitor.network_monitor import NetworkMonitor, measure_round_comm


def run_experiment(dataset_name, device, distribution="IID", alpha=0.5,
                    use_trust=False, cfg=CONFIG):
    """
    Run one complete FL experiment.
    Returns: dict with per-round metrics.
    """
    label = f"{dataset_name} | {'Trust-FedAvg' if use_trust else 'FedAvg'} | {distribution}"
    print(f"\n{'=' * 60}")
    print(f"  Running: {label}")
    print(f"{'=' * 60}")

    # Load data
    train_data, test_data = load_dataset(dataset_name)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    # Partition
    num_clients = cfg["num_clients"]
    if distribution == "IID":
        client_indices = iid_partition(train_data, num_clients)
    else:
        client_indices = noniid_partition(train_data, num_clients, alpha=alpha)

    # Model
    global_model = get_model(dataset_name, device)
    trust_calc = TrustScoreCalculator(num_clients)

    # Clients
    clients = [
        FedClient(i, client_indices[i], train_data, cfg["batch_size"], cfg["learning_rate"], device)
        for i in range(len(client_indices))
    ]
    actual_clients = len(clients)

    # Metrics storage
    history = {
        "accuracy": [], "loss": [], "train_time": [], "comm_kb": [],
        "cpu_percent": [], "ram_mb": [], "trust_scores": [],
    }

    net_monitor = NetworkMonitor()
    model_kb = measure_round_comm(global_model, actual_clients)

    for round_num in range(1, cfg["num_rounds"] + 1):
        net_monitor.reset()
        cpu_before = psutil.cpu_percent(interval=None)
        ram_before = psutil.virtual_memory().used / (1024 ** 2)
        round_start = time.time()

        global_state = {"model": global_model, "state": global_model.state_dict()}

        # Local training
        client_states, client_sizes, client_losses = [], [], []
        for client in clients:
            state, metrics = client.train(global_state, cfg["local_epochs"])
            client_states.append(state)
            client_sizes.append(metrics["num_samples"])
            client_losses.append(metrics["loss"])

        # Trust scores
        round_trusts = []
        if use_trust:
            for i, client in enumerate(clients):
                prev = client.train_losses[-2] if len(client.train_losses) > 1 else None
                trust = trust_calc.compute(i, client_losses[i], client_losses, prev)
                round_trusts.append(trust)
        else:
            round_trusts = [1.0] * actual_clients

        # Aggregate
        if use_trust:
            global_model = trust_fedavg_aggregate(
                global_model, client_states, client_sizes, round_trusts, cfg["trust_threshold"]
            )
        else:
            global_model = fedavg_aggregate(global_model, client_states, client_sizes)

        # Evaluate
        eval_metrics = evaluate(global_model, test_loader, device)

        # System metrics
        cpu_after = psutil.cpu_percent(interval=None)
        ram_after = psutil.virtual_memory().used / (1024 ** 2)
        round_time = time.time() - round_start

        history["accuracy"].append(eval_metrics["accuracy"])
        history["loss"].append(eval_metrics["loss"])
        history["train_time"].append(round_time)
        history["comm_kb"].append(model_kb)
        history["cpu_percent"].append((cpu_before + cpu_after) / 2)
        history["ram_mb"].append(ram_after - ram_before)
        history["trust_scores"].append(np.mean(round_trusts))

        if round_num % 5 == 0 or round_num == 1:
            print(f"  Round {round_num:2d} | Acc: {eval_metrics['accuracy']:.4f} | "
                  f"Loss: {eval_metrics['loss']:.4f} | Time: {round_time:.1f}s | "
                  f"Comm: {model_kb:.1f}KB | Avg Trust: {np.mean(round_trusts):.3f}")

    print(f"\n  Final Accuracy: {history['accuracy'][-1]:.4f}")
    print(f"  Final Loss:     {history['loss'][-1]:.4f}")
    print(f"  Total Comm:     {sum(history['comm_kb']):.1f} KB")
    return history

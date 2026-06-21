"""
Central configuration for all Trust-FedAvg / FedAvg experiments.
Import CONFIG wherever hyperparameters are needed instead of hardcoding values.
"""

CONFIG = {
    "num_clients":     5,
    "num_rounds":      20,
    "local_epochs":    1,
    "batch_size":      32,
    "learning_rate":   0.01,
    "datasets":        ["MNIST", "FashionMNIST", "CIFAR10"],
    "dirichlet_alpha": 0.5,   # Non-IID concentration parameter
    "trust_threshold": 0.1,   # Min trust score required to participate in aggregation
    "seed":            42,
}


def print_config(cfg=CONFIG):
    print("Configuration:")
    for k, v in cfg.items():
        print(f"   {k}: {v}")

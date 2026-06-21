"""
Client data partitioning strategies: IID and Dirichlet-based Non-IID.
"""

import random
import numpy as np
import matplotlib.pyplot as plt


def iid_partition(dataset, num_clients):
    """Split dataset equally and randomly across clients (IID)."""
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    chunk = len(indices) // num_clients
    return [indices[i * chunk:(i + 1) * chunk] for i in range(num_clients)]


def noniid_partition(dataset, num_clients, alpha=0.5):
    """
    Dirichlet-based Non-IID partition.
    alpha: concentration parameter
      - small alpha (e.g. 0.1) => very non-IID (each client gets 1-2 classes)
      - large alpha (e.g. 10)  => nearly IID
    """
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    num_classes = len(np.unique(labels))
    class_indices = {c: np.where(labels == c)[0].tolist() for c in range(num_classes)}

    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx = class_indices[c]
        random.shuffle(idx)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (proportions * len(idx)).astype(int)
        diff = len(idx) - proportions.sum()
        proportions[0] += diff
        start = 0
        for i, count in enumerate(proportions):
            client_indices[i].extend(idx[start:start + count])
            start += count

    client_indices = [c for c in client_indices if len(c) > 0]
    return client_indices


def show_distribution(client_indices, dataset, title="Data Distribution", save_dir="results/plots"):
    """Plot and save class distribution across clients as a stacked bar chart."""
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    num_clients = len(client_indices)
    num_classes = len(np.unique(labels))

    dist = np.zeros((num_clients, num_classes))
    for i, idx in enumerate(client_indices):
        for j in idx:
            dist[i][labels[j]] += 1

    fig, ax = plt.subplots(figsize=(10, 4))
    bottom = np.zeros(num_clients)
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    for c in range(num_classes):
        ax.bar(range(num_clients), dist[:, c], bottom=bottom, color=colors[c], label=f'Class {c}')
        bottom += dist[:, c]

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Client ID")
    ax.set_ylabel("Number of Samples")
    ax.legend(loc='upper right', ncol=5, fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/dist_{title.replace(' ', '_')}.png", dpi=100)
    plt.show()

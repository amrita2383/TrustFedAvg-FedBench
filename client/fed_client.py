"""
FedClient: represents a single federated learning client.
Handles local training (SGD) on its own data shard.
"""

import copy
import time

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset


class FedClient:
    def __init__(self, client_id, indices, dataset, batch_size, lr, device):
        self.client_id = client_id
        self.device = device
        self.loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=True)
        self.lr = lr
        self.prev_loss = None
        self.train_losses = []

    def train(self, global_weights, local_epochs):
        """Train local model, return updated weights + metrics."""
        model = copy.deepcopy(global_weights["model"])
        model.load_state_dict(global_weights["state"])
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        batches = 0
        t_start = time.time()

        for _ in range(local_epochs):
            for X, y in self.loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batches += 1

        avg_loss = total_loss / max(batches, 1)
        train_time = time.time() - t_start
        self.train_losses.append(avg_loss)

        metrics = {
            "loss": avg_loss,
            "train_time": train_time,
            "num_samples": len(self.loader.dataset),
        }
        return model.state_dict(), metrics

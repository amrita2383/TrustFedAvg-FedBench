"""
Global model evaluation on a held-out test set.
"""

import torch.nn as nn
import torch


def evaluate(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = correct = total = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            total_loss += criterion(out, y).item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return {
        "accuracy": correct / total,
        "loss": total_loss / len(test_loader)
    }

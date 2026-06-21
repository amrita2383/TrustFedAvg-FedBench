"""
Aggregation strategies executed on the server.
- fedavg_aggregate:       standard size-weighted FedAvg
- trust_fedavg_aggregate: trust-and-size-weighted Trust-FedAvg
"""

import copy
import torch


def fedavg_aggregate(global_model, client_states, client_sizes):
    """Standard FedAvg: weighted average by dataset size."""
    total = sum(client_sizes)
    averaged = copy.deepcopy(client_states[0])
    for key in averaged:
        averaged[key] = torch.zeros_like(averaged[key], dtype=torch.float32)
        for state, size in zip(client_states, client_sizes):
            averaged[key] += (size / total) * state[key].float()
    global_model.load_state_dict(averaged)
    return global_model


def trust_fedavg_aggregate(global_model, client_states, client_sizes,
                            trust_scores, threshold=0.1):
    """Trust-FedAvg: weighted by trust_score x dataset_size; clients below
    threshold are excluded from this round's aggregation."""
    valid_states = []
    valid_sizes = []
    valid_trusts = []

    for state, size, trust in zip(client_states, client_sizes, trust_scores):
        if trust >= threshold:
            valid_states.append(state)
            valid_sizes.append(size)
            valid_trusts.append(trust)

    if not valid_states:
        return global_model  # fallback: no update

    weights_raw = [t * s for t, s in zip(valid_trusts, valid_sizes)]
    total_w = sum(weights_raw)

    averaged = copy.deepcopy(valid_states[0])
    for key in averaged:
        averaged[key] = torch.zeros_like(averaged[key], dtype=torch.float32)
        for state, w in zip(valid_states, weights_raw):
            averaged[key] += (w / total_w) * state[key].float()

    global_model.load_state_dict(averaged)
    return global_model

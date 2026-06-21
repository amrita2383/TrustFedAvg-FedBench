"""
Trust score computation for Trust-FedAvg.

Ti = clip(0.6 * Qi + 0.4 * Hi, 0.1, 1.0)

Qi: instantaneous quality term, based on a client's loss relative to all clients' losses.
Hi: historical consistency term, an exponential moving average of relative loss improvement.
"""


class TrustScoreCalculator:
    def __init__(self, num_clients):
        self.H = {i: 0.5 for i in range(num_clients)}  # initial historical score

    def compute(self, client_id, loss, all_losses, prev_loss):
        # Quality score: how good is this client's loss vs others
        total = sum(all_losses) + 1e-8
        Qi = 1.0 - (loss / total)
        Qi = max(0.0, min(1.0, Qi))

        # Historical consistency: exponential moving average of loss improvement
        if prev_loss is not None and prev_loss > 0:
            delta = max(0.0, (prev_loss - loss) / (prev_loss + 1e-8))
        else:
            delta = 0.5

        self.H[client_id] = 0.3 * delta + 0.7 * self.H[client_id]
        Hi = self.H[client_id]

        Ti = 0.6 * Qi + 0.4 * Hi
        Ti = max(0.1, min(1.0, Ti))
        return Ti

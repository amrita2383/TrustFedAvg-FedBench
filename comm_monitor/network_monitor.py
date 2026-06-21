"""
Communication overhead estimation and live system network monitoring.
"""

import psutil


class NetworkMonitor:
    """
    Tracks network bytes using psutil.
    In a real distributed setting, this captures actual host traffic.
    In simulation, we estimate bytes = model parameter count x 4 (float32).
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._start = psutil.net_io_counters()

    def get_bytes(self):
        curr = psutil.net_io_counters()
        sent = curr.bytes_sent - self._start.bytes_sent
        received = curr.bytes_recv - self._start.bytes_recv
        return sent, received

    @staticmethod
    def estimate_model_bytes(model):
        """Estimate bytes needed to transmit model weights (float32 = 4 bytes)."""
        total = sum(p.numel() for p in model.parameters())
        return total * 4  # bytes


def measure_round_comm(model, num_clients):
    """
    Estimated communication per round:
    - Server -> each client: model size (download)
    - Each client -> server: model size (upload)
    Returns total bytes in KB.
    """
    model_bytes = NetworkMonitor.estimate_model_bytes(model)
    total = model_bytes * num_clients * 2  # upload + download
    return total / 1024  # KB

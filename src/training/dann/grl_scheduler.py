import math

def compute_grl_lambda(
    epoch: int,
    total_epochs: int,
    max_lambda: float,
    gamma: float,
    enable: bool,
) -> float:

    if not enable:
        return max_lambda

    progress = epoch / total_epochs
    schedule = 2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0
    return max_lambda * schedule
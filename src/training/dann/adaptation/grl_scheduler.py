import math
from src.loaders.config import ConfigModel

class GRLLambdaScheduler:

    def __init__(self, config: ConfigModel):
        self.is_enabled = config.domain_adaptation.grl_scheduler_enable
        self.gradient_reversal_lambda = config.domain_adaptation.gradient_reversal_lambda
        self.gamma = config.domain_adaptation.grl_scheduler_gamma
        self.total_epochs = config.training.epochs

    def get_grl_lambda(self, epoch: int) -> float:
        if not self.is_enabled:
            return self.gradient_reversal_lambda

        progress = epoch / self.total_epochs
        schedule = 2.0 / (1.0 + math.exp(-self.gamma * progress)) - 1.0
        return self.gradient_reversal_lambda * schedule

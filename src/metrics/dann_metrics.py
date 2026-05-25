import torch
from src.metrics.baseline_metrics import BaselineMetrics

class DANNMetrics(BaselineMetrics):

    def __init__(self):
        super().__init__()

        self.domain_loss = 0.0
        self.domain_correct = 0
        self.domain_total = 0
        self.total_loss = 0.0

    def update(self, 
        labels: torch.Tensor,
        domains: torch.Tensor,
        label_logits: torch.Tensor,
        domain_logits: torch.Tensor,
        label_loss: torch.Tensor,
        domain_loss: torch.Tensor,
        total_loss: torch.Tensor,
    ):
        super().update(labels, label_logits, label_loss)

        label_batch_size = labels.size(0)
        domain_batch_size = domains.size(0)

        self.total_loss += total_loss.item() * label_batch_size
        self.domain_loss += domain_loss.item() * domain_batch_size
        
        domain_predictions = torch.argmax(domain_logits, dim=1)
        self.domain_correct += (domain_predictions == domains).sum().item()
        self.domain_total += domain_batch_size

    def get_metrics(self) -> dict[str, float]:

        if self.label_total == 0 or self.domain_total == 0:
            raise ValueError("Cannot evaluate an empty data loader.")

        metrics = super().get_metrics()

        return {
            **metrics,
            "average_loss": self.total_loss / self.label_total,
            "domain_loss": self.domain_loss / self.domain_total,
            "domain_accuracy": self.domain_correct / self.domain_total,
        }

    def reset(self):
        super().reset()

        self.domain_loss = 0.0
        self.domain_correct = 0
        self.domain_total = 0
        self.total_loss = 0.0

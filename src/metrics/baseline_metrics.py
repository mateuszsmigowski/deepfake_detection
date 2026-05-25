import torch
import time

class BaselineMetrics:

    def __init__(self):
        self.label_loss = 0.0
        self.label_correct = 0
        self.label_total = 0
        self.label_scores = []
        self.label_targets = []
        self.epoch_start = time.perf_counter()
        self.epoch_time = 0.0

    def update(
        self,
        labels: torch.Tensor,
        label_logits: torch.Tensor,
        label_loss: torch.Tensor,
    ):
        batch_size = labels.size(0)
        self.label_loss += label_loss.item() * batch_size

        probabilities = torch.sigmoid(label_logits).detach().flatten().cpu()
        targets = labels.detach().flatten().float().cpu()
        predictions = (probabilities >= 0.5).float()

        self.label_correct += (predictions == targets).sum().item()

        self.label_total += batch_size
        self.label_scores.append(probabilities)
        self.label_targets.append(targets)

        self.epoch_time = time.perf_counter() - self.epoch_start

    def get_metrics(self) -> dict[str, float]:

        if self.label_total == 0:
            raise ValueError("Cannot evaluate an empty data loader.")

        targets = torch.cat(self.label_targets)
        scores = torch.cat(self.label_scores)
        predictions = (scores >= 0.5).float()

        positive_count = (targets == 1).sum().item()
        negative_count = (targets == 0).sum().item()

        if positive_count == 0 or negative_count == 0:
            raise ValueError("Cannot compute balanced accuracy for empty targets.")

        true_positive = ((predictions == 1) & (targets == 1)).sum().item()
        true_negative = ((predictions == 0) & (targets == 0)).sum().item()
        false_positive = ((predictions == 1) & (targets == 0)).sum().item()
        false_negative = ((predictions == 0) & (targets == 1)).sum().item()

        sensitivity = true_positive / positive_count
        specificity = true_negative / negative_count
        precision = self._safe_divide(true_positive, true_positive + false_positive)
        recall = self._safe_divide(true_positive, true_positive + false_negative)
        f1_score = self._safe_divide(2 * precision * recall, precision + recall)
        auc_roc = self._binary_auc(scores, targets)

        return {
            "average_loss": self.label_loss / self.label_total,
            "label_loss": self.label_loss / self.label_total,
            "accuracy": self.label_correct / self.label_total,
            "balanced_accuracy": (sensitivity + specificity) / 2,
            "auc": auc_roc,
            "auc_roc": auc_roc,
            "f1_score": f1_score,
            "precision": precision,
            "recall": recall,
            "time": self.epoch_time,
        }

    @staticmethod
    def _safe_divide(numerator: float, denominator: float) -> float:
        if denominator == 0:
            return 0.0
        return numerator / denominator

    @staticmethod
    def _binary_auc(scores: torch.Tensor, targets: torch.Tensor) -> float:

        positive_mask = targets == 1
        positive_count = positive_mask.sum().item()
        negative_count = (targets == 0).sum().item()

        if positive_count == 0 or negative_count == 0:
            raise ValueError("AUC requires both positive and negative samples.")

        order = torch.argsort(scores)
        sorted_scores = scores[order]
        ranks = torch.empty_like(scores, dtype=torch.float32)
        start = 0

        while start < scores.numel():
            end = start + 1
            while end < scores.numel() and sorted_scores[end] == sorted_scores[start]:
                end += 1
            average_rank = (start + 1 + end) / 2.0
            ranks[order[start:end]] = average_rank
            start = end

        positive_rank_sum = ranks[positive_mask].sum().item()
        
        return (
            positive_rank_sum - positive_count * (positive_count + 1) / 2
        ) / (positive_count * negative_count)


    def reset(self):
        self.epoch_start = time.perf_counter()
        self.epoch_time = 0.0
        self.label_loss = 0.0
        self.label_correct = 0
        self.label_total = 0
        self.label_scores = []
        self.label_targets = []

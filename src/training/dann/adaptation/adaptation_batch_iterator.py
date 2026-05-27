import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass

@dataclass
class AdaptationLabeledBatch:
    images: torch.Tensor
    labels: torch.Tensor
    domains: torch.Tensor
    metadata: dict

    def to(self, device: torch.device):
        self.images = self.images.to(device)
        self.labels = self.labels.to(device)
        self.domains = self.domains.to(device)

class AdaptationBatchIterator:

    def __init__(
        self,
        source_label_loader: DataLoader,
        source_domain_loader: DataLoader,
        target_domain_loader: DataLoader,
    ):

        self._source_label_loader = source_label_loader
        self._source_domain_loader = source_domain_loader
        self._target_domain_loader = target_domain_loader
        self._source_label_iter = None
        self._source_domain_iter = None
        self._target_domain_iter = None
        self._steps_per_epoch = len(self._source_label_loader)
        self._current_step = 0

    def __iter__(self):

        self._current_step = 0
        self._source_label_iter = iter(self._source_label_loader)
        self._source_domain_iter = iter(self._source_domain_loader)
        self._target_domain_iter = iter(self._target_domain_loader)
        
        return self

    def __next__(self):

        if self._current_step >= self._steps_per_epoch:
            raise StopIteration

        self._current_step += 1

        return (
            self._adapt(self._next_source_label_batch()),
            self._adapt(self._next_source_domain_batch()),
            self._adapt(self._next_target_domain_batch()),
        )

    @staticmethod
    def _adapt(batch) -> AdaptationLabeledBatch:
        return AdaptationLabeledBatch(
            images=batch["image"],
            labels=batch["label"],
            domains=batch["domain"],
            metadata=batch["metadata"],
        )

    def _next_source_label_batch(self):
        return self._next_batch("_source_label_iter", self._source_label_loader)

    def _next_source_domain_batch(self):
        return self._next_batch("_source_domain_iter", self._source_domain_loader)

    def _next_target_domain_batch(self):
        return self._next_batch("_target_domain_iter", self._target_domain_loader)

    def _next_batch(self, iterator_name: str, loader: DataLoader):
        iterator = getattr(self, iterator_name)
        try:
            return next(iterator)
        except StopIteration:
            iterator = iter(loader)
            setattr(self, iterator_name, iterator)
            return next(iterator)

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

    def __init__(self, source_train_loader: DataLoader, target_train_loader: DataLoader):

        self._source_loader = source_train_loader
        self._target_loader = target_train_loader
        self._source_iter = None
        self._target_iter = None
        self._steps_per_epoch = len(self._source_loader)
        self._current_step = 0

    def __iter__(self):

        self._current_step = 0
        self._source_iter = iter(self._source_loader)
        self._target_iter = iter(self._target_loader)
        
        return self

    def __next__(self):

        if self._current_step >= self._steps_per_epoch:
            raise StopIteration

        self._current_step += 1

        source_batch = self._next_source_batch()
        adapted_source_batch = AdaptationLabeledBatch(
            images=source_batch["image"],
            labels=source_batch["label"],
            domains=source_batch["domain"],
            metadata=source_batch["metadata"],
        )

        target_batch = self._next_target_batch()
        adapted_target_batch = AdaptationLabeledBatch(
            images=target_batch["image"],
            labels=target_batch["label"],
            domains=target_batch["domain"],
            metadata=target_batch["metadata"],
        )

        return adapted_source_batch, adapted_target_batch

    def _next_source_batch(self):
        try:
            return next(self._source_iter)
        except StopIteration:
            self._source_iter = iter(self._source_loader)
            return next(self._source_iter)

    def _next_target_batch(self):
        try:
            return next(self._target_iter)
        except StopIteration:
            self._target_iter = iter(self._target_loader)
            return next(self._target_iter)

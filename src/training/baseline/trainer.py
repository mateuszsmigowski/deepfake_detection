import time
import torch
from torch import nn
from torch.optim import Adam
from src.loaders.config import ConfigModel
from torch.utils.data import DataLoader
from src.training.helpers import resolve_device
from src.metrics.baseline_metrics import BaselineMetrics
from src.metrics.logger import MetricsLogger

class BaselineTrainer:

    def __init__(
        self,
        config: ConfigModel,
        model: nn.Module,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        test_loader: DataLoader,
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.optimizer = Adam(
            model.parameters(),
            lr=config.training.learning_rate,
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.metrics = BaselineMetrics()
        self.logger = MetricsLogger(config)

    def run(self):

        device = resolve_device(self.config.runtime.device)
        self.model.to(device)

        for epoch in range(self.config.training.epochs): 
            
            self.model.train()
            self.metrics.reset()

            for batch in self.train_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                self.optimizer.zero_grad()

                logits = self.model(images)
                loss = self.criterion(logits, labels)

                loss.backward()
                self.optimizer.step()

                self.metrics.update(labels, logits, loss)

            metrics = self.metrics.get_metrics()
            self.logger.log_metrics("train", metrics, epoch + 1)

            if (epoch + 1) % self.config.training.validation_interval == 0:
                metrics = self._evaluate(device, self.validation_loader)
                self.logger.log_metrics("validation", metrics, epoch + 1)
                self.logger.save_best_checkpoint(self.model, metrics, "validation", epoch + 1)
        
        self.logger.load_best_checkpoint(self.model, device)
        metrics = self._evaluate(device, self.test_loader)
        self.logger.log_metrics("test", metrics)

    def _evaluate(self, device: torch.device, loader: DataLoader):
        
        self.model.eval()
        self.metrics.reset()

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                logits = self.model(images)
                loss = self.criterion(logits, labels)

                self.metrics.update(labels, logits, loss)

        return self.metrics.get_metrics()
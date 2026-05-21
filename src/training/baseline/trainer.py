import torch
from torch import nn
from src.loaders.config import ConfigModel
from torch.utils.data import DataLoader
from src.training.helpers import resolve_device, build_optimizer, EarlyStopping, build_label_criterion
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
        self.device = resolve_device(config.runtime.device)
        self.criterion = build_label_criterion(config, self.device)
        self.optimizer = build_optimizer(model, config.training)
        self.metrics = BaselineMetrics()
        self.logger = MetricsLogger(config)
        patience = config.training.early_stopping_patience
        self.early_stopping = EarlyStopping(patience) if patience is not None else None

    def run(self):

        self.model.to(self.device)

        for epoch in range(self.config.training.epochs): 
            
            self.model.train()
            self.metrics.reset()

            for batch in self.train_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                self.optimizer.zero_grad()

                logits = self.model(images)
                loss = self.criterion(logits, labels)

                loss.backward()
                self.optimizer.step()

                self.metrics.update(labels, logits, loss)

            metrics = self.metrics.get_metrics()
            self.logger.log_metrics("train", metrics, epoch + 1)

            if (epoch + 1) % self.config.training.validation_interval == 0:
                metrics = self._evaluate(self.validation_loader)
                self.logger.log_metrics("validation", metrics, epoch + 1)
                improved = self.logger.save_best_checkpoint(self.model, metrics, "validation", epoch + 1)
                if self.early_stopping and self.early_stopping.step(improved):
                    break
        
        self.logger.load_best_checkpoint(self.model, self.device)
        metrics = self._evaluate(self.test_loader)
        self.logger.log_metrics("test", metrics)

    def _evaluate(self, loader: DataLoader):
        
        self.model.eval()
        self.metrics.reset()

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, labels)

                self.metrics.update(labels, logits, loss)

        return self.metrics.get_metrics()
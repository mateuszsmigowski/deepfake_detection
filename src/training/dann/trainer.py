import torch
from src.loaders.config import ConfigModel
from torch.optim import Adam
from torch.utils.data import DataLoader
from src.metrics import DANNMetrics, MetricsLogger, BaselineMetrics
from src.training.helpers import resolve_device

class DANNTrainer:

    def __init__(
        self,
        config: ConfigModel,
        model: torch.nn.Module,
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
        self.label_criterion = torch.nn.BCEWithLogitsLoss()
        self.domain_criterion = torch.nn.CrossEntropyLoss()
        self.metrics = DANNMetrics()
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
                domains = batch["domain"].to(device)

                self.optimizer.zero_grad()

                label_logits, domain_logits = self.model(images)
                label_loss = self.label_criterion(label_logits, labels)
                domain_loss = self.domain_criterion(domain_logits, domains)
                loss = label_loss + self.config.domain_adaptation.domain_loss_weight * domain_loss

                loss.backward()
                self.optimizer.step()

                self.metrics.update(
                    labels,
                    domains,
                    label_logits,
                    domain_logits,
                    label_loss,
                    domain_loss,
                    loss
                )
                
            metrics = self.metrics.get_metrics()
            self.logger.log_metrics("train", metrics, epoch + 1)

            if (epoch + 1) % self.config.training.validation_interval == 0:
                
                metrics = self._evaluate(
                    device,
                    self.validation_loader,
                    include_domain_metrics=True,
                )
                self.logger.log_metrics("validation", metrics, epoch + 1)
                self.logger.save_best_checkpoint(self.model, metrics, "validation", epoch + 1)

        self.logger.load_best_checkpoint(self.model, device)
        metrics = self._evaluate(
            device,
            self.test_loader,
            include_domain_metrics=False,
        )
        self.logger.log_metrics("test", metrics)

    def _evaluate(
        self,
        device: torch.device,
        loader: DataLoader,
        include_domain_metrics: bool,
    ) -> dict[str, float]:

        self.model.eval()
        metrics = DANNMetrics() if include_domain_metrics else BaselineMetrics()

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                label_logits, domain_logits = self.model(images)
                label_loss = self.label_criterion(label_logits, labels)

                if include_domain_metrics:
                    domains = batch["domain"].to(device)
                    domain_loss = self.domain_criterion(domain_logits, domains)
                    loss = label_loss + self.config.domain_adaptation.domain_loss_weight * domain_loss

                    metrics.update(
                        labels,
                        domains,
                        label_logits,
                        domain_logits,
                        label_loss,
                        domain_loss,
                        loss,
                    )
                else:
                    metrics.update(
                        labels,
                        label_logits,
                        label_loss,
                    )

        return metrics.get_metrics()
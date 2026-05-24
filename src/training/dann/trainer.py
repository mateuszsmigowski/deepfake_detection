import torch
from src.loaders.config import ConfigModel
from torch.utils.data import DataLoader
from src.metrics import DANNMetrics, MetricsLogger, BaselineMetrics
from src.training.helpers import resolve_device, build_optimizer, EarlyStopping, build_label_criterion
from src.training.dann.grl_scheduler import compute_grl_lambda

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
        self.device = resolve_device(config.runtime.device)
        self.label_criterion = build_label_criterion(config, self.device)
        self.optimizer = build_optimizer(model, config.training)
        self.domain_criterion = torch.nn.CrossEntropyLoss()
        self.metrics = DANNMetrics()
        self.logger = MetricsLogger(config)
        patience = config.training.early_stopping_patience
        self.early_stopping = EarlyStopping(patience) if patience is not None else None

    def run(self):
        
        self.model.to(self.device)

        for epoch in range(self.config.training.epochs):
            
            grl_lambda = compute_grl_lambda(
                epoch,
                self.config.training.epochs,
                self.config.domain_adaptation.gradient_reversal_lambda,
                self.config.domain_adaptation.grl_scheduler_gamma,
                self.config.domain_adaptation.grl_scheduler_enable,
            )
            self.model.train()
            self.metrics.reset()
        
            for batch in self.train_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                domains = batch["domain"].to(self.device)

                self.optimizer.zero_grad()

                label_logits, domain_logits, _ = self.model(images, grl_lambda)
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
                    self.validation_loader,
                    include_domain_metrics=True,
                    grl_lambda=grl_lambda,
                )
                self.logger.log_metrics("validation", metrics, epoch + 1)
                improved = self.logger.save_best_checkpoint(self.model, metrics, "validation", epoch + 1)
                if self.early_stopping and self.early_stopping.step(improved):
                    break

        self.logger.load_best_checkpoint(self.model, self.device)
        metrics = self._evaluate(
            self.test_loader,
            include_domain_metrics=False,
        )
        self.logger.log_metrics("test", metrics)

    def _evaluate(
        self,
        loader: DataLoader,
        include_domain_metrics: bool,
        grl_lambda: float | None = None,
    ) -> dict[str, float]:

        self.model.eval()
        metrics = DANNMetrics() if include_domain_metrics else BaselineMetrics()

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                label_logits, domain_logits, _ = self.model(images, grl_lambda)
                label_loss = self.label_criterion(label_logits, labels)

                if include_domain_metrics:
                    domains = batch["domain"].to(self.device)
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
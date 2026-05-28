import torch
from src.loaders.config import ConfigModel
from torch.utils.data import DataLoader
from src.metrics import DANNMetrics, MetricsLogger, BaselineMetrics, FeatureVisualizer
from src.training.helpers import resolve_device, build_optimizer, EarlyStopping, build_label_criterion
from src.training.dann.adaptation.grl_scheduler import GRLLambdaScheduler

class DANNGeneralizationTrainer:

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
        self.feature_visualizer = FeatureVisualizer(self.logger.run_dir, 5)
        patience = config.training.early_stopping_patience
        self.early_stopping = EarlyStopping(patience) if patience is not None else None
        self.grl_lambda_scheduler = GRLLambdaScheduler(config)

    def run(self):
        
        self.model.to(self.device)

        for epoch in range(self.config.training.epochs):
            
            epoch_number = epoch + 1
            grl_lambda = self.grl_lambda_scheduler.get_grl_lambda(epoch)
            self.model.train()
            self.metrics.reset()
            self.feature_visualizer.reset()
        
            for batch in self.train_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                domains = batch["domain"].to(self.device)

                self.optimizer.zero_grad()

                domain_mask = labels == 1
                output = self.model(
                    images,
                    grl_lambda,
                    domain_mask=domain_mask,
                )
                label_loss = self.label_criterion(output.label_logits, labels)
                domain_labels = domains[domain_mask]

                if domain_labels.numel() > 0:
                    domain_loss = self.domain_criterion(output.domain_logits, domain_labels)
                    loss = label_loss + self.config.domain_adaptation.domain_loss_weight * domain_loss
                else:
                    loss = label_loss

                loss.backward()
                self.optimizer.step()

                if domain_labels.numel() > 0:
                    self.metrics.update(
                        labels,
                        domain_labels,
                        output.label_logits,
                        output.domain_logits,
                        label_loss,
                        domain_loss,
                        loss
                    )
                else:
                    self.metrics.update_label_only(
                        labels,
                        output.label_logits,
                        label_loss,
                        loss,
                    )

                self.feature_visualizer.add_features(
                    epoch_number,
                    output.features,
                    labels,
                    batch["metadata"]["domain"],
                )
                
            metrics = self.metrics.get_metrics()
            self.logger.log_metrics("train", metrics, epoch_number)
            self.feature_visualizer.flush(epoch_number, "train")

            if epoch_number % self.config.training.validation_interval == 0:
                
                metrics = self._evaluate(
                    self.validation_loader,
                    include_domain_metrics=True,
                    grl_lambda=grl_lambda,
                )
                self.logger.log_metrics("validation", metrics, epoch_number)
                improved = self.logger.save_best_checkpoint(self.model, metrics, "validation", epoch_number)
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

                domain_mask = labels == 1
                output = self.model(
                    images,
                    grl_lambda,
                    domain_mask=domain_mask,
                    skip_domain=not include_domain_metrics,
                )
                label_loss = self.label_criterion(output.label_logits, labels)

                if include_domain_metrics:
                    domains = batch["domain"].to(self.device)
                    domain_labels = domains[domain_mask]

                    if domain_labels.numel() > 0:
                        domain_loss = self.domain_criterion(output.domain_logits, domain_labels)
                        loss = label_loss + self.config.domain_adaptation.domain_loss_weight * domain_loss

                        metrics.update(
                            labels,
                            domain_labels,
                            output.label_logits,
                            output.domain_logits,
                            label_loss,
                            domain_loss,
                            loss,
                        )
                    else:
                        metrics.update_label_only(
                            labels,
                            output.label_logits,
                            label_loss,
                            label_loss,
                        )
                else:
                    metrics.update(
                        labels,
                        output.label_logits,
                        label_loss,
                    )

        return metrics.get_metrics()

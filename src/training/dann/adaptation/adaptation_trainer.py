import torch
from src.loaders.config import ConfigModel
from torch.utils.data import DataLoader
from src.metrics import DANNMetrics, MetricsLogger, BaselineMetrics, FeatureVisualizer
from src.training.helpers import resolve_device, build_optimizer, EarlyStopping, build_label_criterion
from src.training.dann.adaptation.grl_scheduler import GRLLambdaScheduler
from src.training.dann.adaptation.adaptation_batch_iterator import AdaptationBatchIterator

class DANNAdaptationTrainer:

    def __init__(
        self,
        config: ConfigModel,
        model: torch.nn.Module,
        source_train_loader: DataLoader,
        target_train_loader: DataLoader,
        validation_loader: DataLoader,
        test_loader: DataLoader,
    ):

        self.config = config
        self.model = model
        self.source_train_loader = source_train_loader
        self.target_train_loader = target_train_loader
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
        self.batch_iterator = AdaptationBatchIterator(source_train_loader, target_train_loader)

    def run(self):
        
        self.model.to(self.device)

        for epoch in range(self.config.training.epochs):
            
            grl_lambda = self.grl_lambda_scheduler.get_grl_lambda(epoch)
            self.model.train()
            self.metrics.reset()
            self.feature_visualizer.reset()
        
            for source_batch, target_batch in self.batch_iterator:

                source_batch.to(self.device)
                target_batch.to(self.device)

                self.optimizer.zero_grad()

                images = torch.cat([source_batch.images, target_batch.images], dim=0)
                labels = torch.cat([source_batch.labels, target_batch.labels], dim=0)
                domains = torch.cat([source_batch.domains, target_batch.domains], dim=0)
                source_mask = torch.cat([
                    torch.ones_like(source_batch.labels, dtype=torch.bool),
                    torch.zeros_like(target_batch.labels, dtype=torch.bool),
                ])
                domain_mask = labels == 1
                domain_labels = domains[domain_mask]

                output = self.model(images, grl_lambda, domain_mask=domain_mask)

                source_label_logits = output.label_logits[source_mask]
                source_labels = labels[source_mask]
                label_loss = self.label_criterion(source_label_logits, source_labels)

                if domain_labels.numel() > 0:
                    domain_loss = self.domain_criterion(output.domain_logits, domain_labels)
                    loss = label_loss + self.config.domain_adaptation.domain_loss_weight * domain_loss
                else:
                    domain_loss = None
                    loss = label_loss

                loss.backward()
                self.optimizer.step()

                if domain_loss is not None:
                    self.metrics.update(
                        source_labels,
                        domain_labels,
                        source_label_logits,
                        output.domain_logits,
                        label_loss,
                        domain_loss,
                        loss,
                    )
                else:
                    self.metrics.update_label_only(
                        source_labels,
                        source_label_logits,
                        label_loss,
                        loss,
                    )

                self.feature_visualizer.add_features(
                    epoch,
                    output.features,
                    labels,
                    [*self._extract_domain_names(source_batch), *self._extract_domain_names(target_batch)],
                )

            metrics = self.metrics.get_metrics()
            self.logger.log_metrics("train", metrics, epoch + 1)
            self.feature_visualizer.flush(epoch, "train")
                
            if (epoch + 1) % self.config.training.validation_interval == 0:
                metrics = self._evaluate(self.validation_loader, include_domain_metrics=False)
                self.logger.log_metrics("validation", metrics, epoch + 1)
                improved = self.logger.save_best_checkpoint(self.model, metrics, "validation", epoch + 1)
                if self.early_stopping and self.early_stopping.step(improved):
                    break

        self.logger.load_best_checkpoint(self.model, self.device)
        metrics = self._evaluate(self.test_loader, include_domain_metrics=False)
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

    def _extract_domain_names(self, batch) -> list[str]:
        return batch.metadata["domain"]

    def _extract_selected_domain_names(self, batch, indices: torch.Tensor) -> list[str]:
        domain_names = self._extract_domain_names(batch)
        return [domain_names[index] for index in indices.cpu().tolist()]

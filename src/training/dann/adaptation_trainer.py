import torch
from src.loaders.config import ConfigModel
from torch.utils.data import DataLoader
from src.metrics import DANNMetrics, MetricsLogger, BaselineMetrics, FeatureVisualizer
from src.training.helpers import resolve_device, build_optimizer, EarlyStopping, build_label_criterion
from src.training.dann.grl_scheduler import compute_grl_lambda

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
        self.feature_visualizer = FeatureVisualizer(self.logger.run_dir)
        self.feature_visualization_interval = 5
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
            if epoch % self.feature_visualization_interval == 0:
                self.feature_visualizer.reset()

            source_iter = iter(self.source_train_loader)
            target_iter = iter(self.target_train_loader)
            steps_per_epoch = len(self.source_train_loader)
        
            for _ in range(steps_per_epoch):

                try:
                    source_batch = next(source_iter)
                except StopIteration:
                    source_iter = iter(self.source_train_loader)
                    source_batch = next(source_iter)

                try:
                    target_batch = next(target_iter)
                except StopIteration:
                    target_iter = iter(self.target_train_loader)
                    target_batch = next(target_iter)

                source_images = source_batch["image"].to(self.device)
                source_labels = source_batch["label"].to(self.device)
                target_images = target_batch["image"].to(self.device)
                target_labels = target_batch["label"].to(self.device)

                self.optimizer.zero_grad()

                source_fake_indices = torch.nonzero(source_labels == 1, as_tuple=False).flatten()
                domain_batch_size = min(source_fake_indices.numel(), target_images.size(0))
                source_domain_mask = torch.zeros_like(source_labels, dtype=torch.bool)
                target_indices = torch.empty(0, dtype=torch.long, device=self.device)

                if domain_batch_size > 0:
                    source_indices = source_fake_indices[
                        torch.randperm(source_fake_indices.numel(), device=self.device)[:domain_batch_size]
                    ]
                    target_indices = torch.randperm(
                        target_images.size(0),
                        device=self.device,
                    )[:domain_batch_size]
                    source_domain_mask[source_indices] = True

                source_label_logits, source_domain_logits, source_features = self.model(
                    source_images,
                    grl_lambda,
                    domain_mask=source_domain_mask,
                )
                selected_target_images = target_images[target_indices]

                source_label_loss = self.label_criterion(source_label_logits, source_labels)

                if domain_batch_size > 0:
                    _, target_domain_logits, target_features = self.model(selected_target_images, grl_lambda)
                    source_domain_labels = torch.zeros(
                        source_domain_logits.size(0),
                        dtype=torch.long,
                        device=self.device,
                    )
                    target_domain_labels = torch.ones(
                        target_domain_logits.size(0),
                        dtype=torch.long,
                        device=self.device,
                    )

                    domain_logits = torch.cat([source_domain_logits, target_domain_logits], dim=0)
                    domain_labels = torch.cat([source_domain_labels, target_domain_labels], dim=0)
                    domain_loss = self.domain_criterion(domain_logits, domain_labels)
                    loss = source_label_loss + self.config.domain_adaptation.domain_loss_weight * domain_loss
                else:
                    target_features = None
                    loss = source_label_loss

                loss.backward()
                self.optimizer.step()

                if domain_batch_size > 0:
                    self.metrics.update(
                        source_labels,
                        domain_labels,
                        source_label_logits,
                        domain_logits,
                        source_label_loss,
                        domain_loss,
                        loss
                    )
                else:
                    self.metrics.update_label_only(
                        source_labels,
                        source_label_logits,
                        source_label_loss,
                        loss,
                    )

                if epoch % self.feature_visualization_interval == 0:
                    self.feature_visualizer.add_batch(
                        source_features, source_labels, self._extract_domain_names(source_batch)
                    )
                    if target_features is not None:
                        self.feature_visualizer.add_batch(
                            target_features,
                            target_labels[target_indices],
                            self._extract_selected_domain_names(target_batch, target_indices),
                        )
                
            metrics = self.metrics.get_metrics()
            self.logger.log_metrics("train", metrics, epoch + 1)

            if epoch % self.feature_visualization_interval == 0:
                output_path = self.feature_visualizer.flush(epoch + 1, "train")
                if output_path is not None:
                    print(f"Feature visualization saved to {output_path}")

            if (epoch + 1) % self.config.training.validation_interval == 0:
                
                metrics = self._evaluate(
                    self.validation_loader,
                    include_domain_metrics=False,
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

                label_logits, _, _ = self.model(
                    images,
                    grl_lambda,
                    skip_domain=not include_domain_metrics,
                )
                label_loss = self.label_criterion(label_logits, labels)

                # if include_domain_metrics:
                #     domains = batch["domain"].to(self.device)
                #     domain_loss = self.domain_criterion(domain_logits, domains)
                #     loss = label_loss + self.config.domain_adaptation.domain_loss_weight * domain_loss

                #     metrics.update(
                #         labels,
                #         domains,
                #         label_logits,
                #         domain_logits,
                #         label_loss,
                #         domain_loss,
                #         loss,
                #     )
                # else:
                metrics.update(
                    labels,
                    label_logits,
                    label_loss,
                )

        return metrics.get_metrics()

    def _extract_domain_names(self, batch: dict) -> list[str]:
        return batch["metadata"]["domain"]

    def _extract_selected_domain_names(self, batch: dict, indices: torch.Tensor) -> list[str]:
        domain_names = self._extract_domain_names(batch)
        return [domain_names[index] for index in indices.cpu().tolist()]

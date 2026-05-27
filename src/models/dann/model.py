from dataclasses import dataclass
import torch
from torch import nn
from src.loaders.config import ModelConfig, DomainAdaptationConfig
from src.models.builder import ModelBuilder
from src.models.dann.gradient_reversal import GradientReversalLayer


class DANNClassifier(nn.Module):

    @dataclass
    class ModelOutput:
        label_logits: torch.Tensor
        domain_logits: torch.Tensor
        features: torch.Tensor

    def __init__(
        self,
        model_config: ModelConfig,
        domain_adaptation_config: DomainAdaptationConfig,
        domains_count: int,
    ):
        super().__init__()

        self.domain_adaptation_config = domain_adaptation_config
        self.domains_count = domains_count
        self.backbone, feature_dim = self._build_backbone(model_config)
        self.gradient_reversal_layer = GradientReversalLayer()
        self.label_classifier = self._build_label_classifier(feature_dim)
        self.domain_classifier = self._build_domain_classifier(feature_dim, domains_count)

    @staticmethod
    def _build_backbone(model_config: ModelConfig) -> tuple[nn.Module, int]:
        return ModelBuilder.build_backbone(model_config)

    @staticmethod
    def _build_label_classifier(feature_dim: int) -> nn.Module:
        
        return nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    @staticmethod
    def _build_domain_classifier(feature_dim: int, domains_count: int) -> nn.Module:

        return nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, domains_count),
        )

    def forward(
        self,
        x: torch.Tensor,
        grl_lambda: float | None = None,
        domain_mask: torch.Tensor | None = None,
        skip_domain: bool = False,
    ) -> ModelOutput:

        features = self.backbone(x)
        label_logits = self.label_classifier(features).squeeze(dim=1)

        if skip_domain:
            domain_logits = features.new_empty((0, self.domains_count))
        else:
            lambda_ = (
                grl_lambda
                if grl_lambda is not None
                else self.domain_adaptation_config.gradient_reversal_lambda
            )
            domain_features = features if domain_mask is None else features[domain_mask]
            reversed_features = self.gradient_reversal_layer(domain_features, lambda_)
            domain_logits = self.domain_classifier(reversed_features)

        return self.ModelOutput(label_logits, domain_logits, features)

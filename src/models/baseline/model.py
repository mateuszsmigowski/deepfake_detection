import torch
from torch import nn
from src.loaders.config import ModelConfig
from src.models.builder import ModelBuilder


class BaselineClassifier(nn.Module):

    def __init__(self, model_config: ModelConfig):
        super().__init__()

        self.backbone, in_features = ModelBuilder.build_backbone(model_config)
        self.label_classifier = self._build_label_classifier(in_features)

    @staticmethod
    def _build_label_classifier(feature_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        label_logits = self.label_classifier(features).squeeze(dim=1)
        return label_logits
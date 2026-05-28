from dataclasses import dataclass
import torch
from torch import nn
from src.loaders.config import ModelConfig
from src.models.builder import ModelBuilder


class BaselineClassifier(nn.Module):

    @dataclass
    class ModelOutput:
        label_logits: torch.Tensor
        features: torch.Tensor

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
        return self.forward_with_features(x).label_logits

    def forward_with_features(self, x: torch.Tensor) -> ModelOutput:
        features = self.backbone(x)
        label_logits = self.label_classifier(features).squeeze(dim=1)
        return self.ModelOutput(label_logits, features)

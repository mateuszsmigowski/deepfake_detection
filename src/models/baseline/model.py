import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from src.loaders.config import ModelConfig


class BaselineClassifier(nn.Module):

    def __init__(self, model_config: ModelConfig):
        super().__init__()

        self.backbone, in_features = self._build_backbone(model_config)
        self.label_classifier = self._build_label_classifier(in_features)

    @staticmethod
    def _build_backbone(model_config: ModelConfig) -> tuple[nn.Module, int]:
        
        match model_config.architecture:
            case ModelConfig.Architecture.RESNET18:
                weights = ResNet18_Weights.DEFAULT if model_config.pretrained else None
                model = resnet18(weights=weights)
            case _:
                raise ValueError(f"Unsupported model architecture: {model_config.architecture}")
        
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        
        if model_config.freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        
        return model, feature_dim

    @staticmethod
    def _build_label_classifier(feature_dim: int) -> nn.Module:
        return nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        label_logits = self.label_classifier(features).squeeze(dim=1)
        return label_logits
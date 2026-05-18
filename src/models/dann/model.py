import torch
from torch import nn
from src.loaders.config import ModelConfig, DomainAdaptationConfig
from torchvision.models import resnet18, ResNet18_Weights
from src.models.dann.gradient_reversal import GradientReversalLayer


class DANNClassifier(nn.Module):

    def __init__(self, model_config: ModelConfig, domain_adaptation_config: DomainAdaptationConfig, domains_count: int = 6):
        super().__init__()

        self.domain_adaptation_config = domain_adaptation_config
        self.backbone, feature_dim = self._build_backbone(model_config)
        self.gradient_reversal_layer = GradientReversalLayer()
        self.label_classifier = self._build_label_classifier(feature_dim)
        self.domain_classifier = self._build_domain_classifier(feature_dim, domains_count)

    @staticmethod
    def _build_backbone(model_config: ModelConfig) -> tuple[nn.Module, int]:

        model = None
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

    @staticmethod
    def _build_domain_classifier(feature_dim: int, domains_count: int) -> nn.Module:

        return nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, domains_count),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        features = self.backbone(x)
        label_logits = self.label_classifier(features).squeeze(dim=1)
        reversed_features = self.gradient_reversal_layer(
            features,
            self.domain_adaptation_config.gradient_reversal_lambda,
        )
        domain_logits = self.domain_classifier(reversed_features)
        return label_logits, domain_logits
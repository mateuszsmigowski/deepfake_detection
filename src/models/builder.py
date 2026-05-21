from torch import nn
from src.loaders.config import ModelConfig
from torchvision.transforms import v2
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    efficientnet_b0,
    EfficientNet_B0_Weights,
    resnet50,
    ResNet50_Weights,
    convnext_tiny,
    ConvNeXt_Tiny_Weights,
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)

_ARCHITECTURE_WEIGHTS = {
    ModelConfig.Architecture.RESNET18: ResNet18_Weights,
    ModelConfig.Architecture.RESNET50: ResNet50_Weights,
    ModelConfig.Architecture.EFFICIENTNET_B0: EfficientNet_B0_Weights,
    ModelConfig.Architecture.CONVNEXT_TINY: ConvNeXt_Tiny_Weights,
    ModelConfig.Architecture.MOBILENET_V3_SMALL: MobileNet_V3_Small_Weights,
}

class ModelBuilder:

    @staticmethod
    def build_backbone(model_config: ModelConfig) -> tuple[nn.Module, int]:

        match model_config.architecture:
            case ModelConfig.Architecture.RESNET18:
                weights = ResNet18_Weights.DEFAULT if model_config.pretrained else None
                model = resnet18(weights=weights)
                feature_dim = model.fc.in_features
                model.fc = nn.Identity()
            case ModelConfig.Architecture.EFFICIENTNET_B0:
                weights = EfficientNet_B0_Weights.DEFAULT if model_config.pretrained else None
                model = efficientnet_b0(weights=weights)
                feature_dim = model.classifier[1].in_features
                model.classifier[1] = nn.Identity()
            case ModelConfig.Architecture.RESNET50:
                weights = ResNet50_Weights.DEFAULT if model_config.pretrained else None
                model = resnet50(weights=weights)
                feature_dim = model.fc.in_features
                model.fc = nn.Identity()
            case ModelConfig.Architecture.CONVNEXT_TINY:
                weights = ConvNeXt_Tiny_Weights.DEFAULT if model_config.pretrained else None
                model = convnext_tiny(weights=weights)
                feature_dim = model.classifier[2].in_features
                model.classifier[2] = nn.Identity()
            case ModelConfig.Architecture.MOBILENET_V3_SMALL:
                weights = MobileNet_V3_Small_Weights.DEFAULT if model_config.pretrained else None
                model = mobilenet_v3_small(weights=weights)
                feature_dim = model.classifier[3].in_features
                model.classifier[3] = nn.Identity()
            case _:
                raise ValueError(f"Unsupported model architecture: {model_config.architecture}")

        if model_config.freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        return model, feature_dim

    @staticmethod
    def get_transforms(architecture: ModelConfig.Architecture) -> v2.Transform:
        return ModelBuilder.get_evaluation_transforms(architecture)

    @staticmethod
    def get_evaluation_transforms(architecture: ModelConfig.Architecture) -> v2.Transform:

        weights = _ARCHITECTURE_WEIGHTS[architecture]
        if weights is None:
            raise ValueError(f"No weights found for architecture: {architecture}")
        
        return weights.DEFAULT.transforms()

    @staticmethod
    def get_training_transforms(architecture: ModelConfig.Architecture) -> v2.Transform:

        return v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0),
            ModelBuilder.get_evaluation_transforms(architecture),
        ])
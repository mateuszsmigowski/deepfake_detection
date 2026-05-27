import yaml
from typing import Any
from pathlib import Path
from .config_model import (
    ConfigModel,
    ExperimentConfig,
    DataConfig,
    RuntimeConfig,
    ModelConfig,
    ManifestConfig,
    TrainingConfig,
    SplitConfig,
    DomainAdaptationConfig,
    OutputsConfig,
)

def load_config(path: Path) -> ConfigModel:

    path = path.resolve()
    with path.open("r", encoding="utf-8") as file:
        values = yaml.safe_load(file)
    if not isinstance(values, dict):
        raise ValueError("Configuration file must contain a YAML mapping.")
    return _config_from_dict(values, path.parent)

def _config_from_dict(values: dict[str, Any], base_path: Path) -> ConfigModel:

    return ConfigModel(
        experiment=_experiment_config(values["experiment"]),
        data=_data_config(values["data"], base_path),
        runtime=_runtime_config(values["runtime"]),
        model=_model_config(values["model"]),
        manifest=_manifest_config(values["manifest"], base_path),
        training=_training_config(values["training"]),
        split=_split_config(values["split"], base_path),
        domain_adaptation=_domain_adaptation_config(values["domain_adaptation"]),
        outputs=_outputs_config(values["outputs"], base_path),
    )

def _experiment_config(values: dict[str, Any]) -> ExperimentConfig:

    def _experiment_mode(value: str) -> ExperimentConfig.Mode:
        try:
            return ExperimentConfig.Mode(value)
        except ValueError:
            raise ValueError(f"Invalid experiment mode: {value}")

    return ExperimentConfig(
        name=values["name"],
        mode=_experiment_mode(values["mode"]),
        protocol=ExperimentConfig.Protocol(values.get("protocol", "generalization")),
        real_domain=values["real_domain"],
        fake_domains=values["fake_domains"],
        held_out_domain=values["held_out_domain"],
    )

def _data_config(values: dict[str, Any], base_path: Path) -> DataConfig:

    return DataConfig(
        dataset_path=base_path / values["dataset_path"],
        train_source_real=int(values.get("train_source_real", 0)),
        train_source_fake=int(values.get("train_source_fake", 0)),
        val_source_real=int(values.get("val_source_real", 0)),
        val_source_fake=int(values.get("val_source_fake", 0)),
        train_target_fake=int(values.get("train_target_fake", 0)),
    )

def _runtime_config(values: dict[str, Any]) -> RuntimeConfig:

    def _runtime_device(value: str) -> RuntimeConfig.Device:
        try:
            return RuntimeConfig.Device(value)
        except ValueError:
            raise ValueError(f"Invalid runtime device: {value}")

    return RuntimeConfig(
        seed=values["seed"],
        device=_runtime_device(values["device"]),
        deterministic=values["deterministic"],
        num_workers=values["num_workers"],
    )

def _model_config(values: dict[str, Any]) -> ModelConfig:

    def _model_architecture(value: str) -> ModelConfig.Architecture:
        try:
            return ModelConfig.Architecture(value)
        except ValueError:
            raise ValueError(f"Invalid model architecture: {value}")

    return ModelConfig(
        architecture=_model_architecture(values["architecture"]),
        pretrained=values["pretrained"],
        freeze_backbone=values["freeze_backbone"],
    )

def _manifest_config(values: dict[str, Any], base_path: Path) -> ManifestConfig:

    return ManifestConfig(
        path=base_path / values["path"],
    )

def _training_config(values: dict[str, Any]) -> TrainingConfig:

    patience = values.get("early_stopping_patience", None)

    return TrainingConfig(
        epochs=values["epochs"],
        batch_size=values["batch_size"],
        learning_rate=values["learning_rate"],
        weight_decay=values["weight_decay"],
        validation_interval=values["validation_interval"],
        early_stopping_patience=patience,
        pos_weight=bool(values.get("pos_weight", False)),
    )

def _split_config(values: dict[str, Any], base_path: Path) -> SplitConfig:

    return SplitConfig(
        train_ratio=values["train_ratio"],
        validation_ratio=values["validation_ratio"],
        test_ratio=values["test_ratio"],
        seed=values["seed"],
        path=base_path / values["path"],
    )

def _domain_adaptation_config(values: dict[str, Any]) -> DomainAdaptationConfig:
    domain_batch_size = values.get("domain_batch_size", None)

    return DomainAdaptationConfig(
        gradient_reversal_lambda=values["gradient_reversal_lambda"],
        domain_loss_weight=values["domain_loss_weight"],
        grl_scheduler_enable=bool(values.get("grl_scheduler_enable", False)),
        grl_scheduler_gamma=float(values.get("grl_scheduler_gamma", 10.0)),
        domain_batch_size=(
            int(domain_batch_size)
            if domain_batch_size is not None
            else None
        ),
    )

def _outputs_config(values: dict[str, Any], base_path: Path) -> OutputsConfig:

    return OutputsConfig(
        runs_path=base_path / values["runs_path"],
        save_checkpoints=values["save_checkpoints"],
        save_metrics=values["save_metrics"],
        checkpoint_metric=values["checkpoint_metric"],
    )

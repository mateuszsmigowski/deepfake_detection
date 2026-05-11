from dataclasses import dataclass
from pathlib import Path
from enum import StrEnum


# MARK: - ConfigModel
@dataclass(frozen=True)
class ConfigModel:
    experiment: ExperimentConfig
    data: DataConfig
    runtime: RuntimeConfig
    model: ModelConfig
    manifest: ManifestConfig
    training: TrainingConfig
    split: SplitConfig
    domain_adaptation: DomainAdaptationConfig
    outputs: OutputsConfig

# MARK: - ExperimentConfig
@dataclass(frozen=True)
class ExperimentConfig:

    class Mode(StrEnum):
        BASELINE = "baseline"
        DANN = "dann"

    name: str
    mode: Mode
    real_domain: str
    fake_domains: list[str]
    held_out_domain: str

# MARK: - DataConfig
@dataclass(frozen=True)
class DataConfig:
    dataset_path: Path

# MARK: - RuntimeConfig
@dataclass(frozen=True)
class RuntimeConfig:

    class Device(StrEnum):
        AUTO = "auto"
        CPU = "cpu"
        CUDA = "cuda"
        MPS = "mps"

    seed: int
    device: Device
    deterministic: bool
    num_workers: int

# MARK: - ModelConfig
@dataclass(frozen=True)
class ModelConfig:

    class  Architecture(StrEnum):
        RESNET18 = "resnet18"

    architecture: Architecture
    pretrained: bool
    freeze_backbone: bool

# MARK: - ManifestConfig
@dataclass(frozen=True)
class ManifestConfig:
    path: Path

# MARK: - SplitConfig
@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float
    validation_ratio: float
    test_ratio: float
    seed: int
    path: Path

# MARK: - TrainingConfig
@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    validation_interval: int

# MARK: - DomainAdaptationConfig
@dataclass(frozen=True)
class DomainAdaptationConfig:
    gradient_reversal_lambda: float
    domain_loss_weight: float

# MARK: - OutputsConfig
@dataclass(frozen=True)
class OutputsConfig:
    runs_path: Path
    save_checkpoints: bool
    save_metrics: bool
    checkpoint_metric: str
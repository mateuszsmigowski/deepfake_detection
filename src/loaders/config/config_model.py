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

    class Protocol(StrEnum):
        BASELINE = "baseline"
        GENERALIZATION = "generalization"
        ADAPTATION = "adaptation"

    name: str
    mode: Mode
    protocol: Protocol
    real_domain: str
    fake_domains: list[str]
    held_out_domain: str

    @property
    def domains(self) -> list[str]:
        return [self.real_domain, *self.fake_domains]

    @property
    def domain_to_int(self) -> dict[str, int]:
        return {domain: index for index, domain in enumerate(self.domains)}

    @property
    def source_domains(self) -> list[str]:
        return [self.real_domain, *[
            domain
            for domain in self.fake_domains
            if domain != self.held_out_domain
        ],]

    @property
    def source_domain_to_int(self) -> dict[str, int]:
        return {
            domain: self.fake_domain_to_int[domain]
            for domain in self.source_domains
        }
    
    @property
    def target_domain_to_int(self) -> dict[str, int]:
        return {self.held_out_domain: self.fake_domain_to_int[self.held_out_domain]}

    @property
    def fake_domain_to_int(self) -> dict[str, int]:
        domain_to_int = {
            domain: index
            for index, domain in enumerate(self.fake_domains)
        }
        domain_to_int[self.real_domain] = -1
        return domain_to_int

# MARK: - DataConfig
@dataclass(frozen=True)
class DataConfig:
    dataset_path: Path
    train_source_real: int
    train_source_fake: int
    val_source_real: int
    val_source_fake: int
    train_target_fake: int

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
        RESNET50 = "resnet50"
        EFFICIENTNET_B0 = "efficientnet_b0"
        CONVNEXT_TINY = "convnext_tiny"
        MOBILENET_V3_SMALL = "mobilenet_v3_small"

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
    weight_decay: float
    validation_interval: int
    early_stopping_patience: int | None = None
    pos_weight: bool = False

# MARK: - DomainAdaptationConfig
@dataclass(frozen=True)
class DomainAdaptationConfig:
    gradient_reversal_lambda: float
    domain_loss_weight: float
    grl_scheduler_enable: bool = False
    grl_scheduler_gamma: float = 10.0

# MARK: - OutputsConfig
@dataclass(frozen=True)
class OutputsConfig:
    runs_path: Path
    save_checkpoints: bool
    save_metrics: bool
    checkpoint_metric: str

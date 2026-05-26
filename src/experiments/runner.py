from dataclasses import dataclass, replace
from src.loaders.config import ConfigModel, ExperimentConfig, ExperimentMode, ModelConfig
from src.orchestrator import orchestrate

@dataclass(frozen=True)
class DANNVariant:
    name: str
    gradient_reversal_lambda: float | None = None
    domain_loss_weight: float | None = None
    freeze_backbone: bool | None = None

SEEDS = [123, 948, 2024]

MODEL_ARCHITECTURES = [
    ModelConfig.Architecture.RESNET18,
    ModelConfig.Architecture.RESNET50,
    ModelConfig.Architecture.EFFICIENTNET_B0,
]

PROTOCOLS = [
    ExperimentConfig.Protocol.GENERALIZATION,
    ExperimentConfig.Protocol.ADAPTATION,
]

DANN_ABLATIONS = [
    DANNVariant("dann-main"),
    # DANNVariant("grl-0p1", gradient_reversal_lambda=0.1),
    # DANNVariant("grl-0p5", gradient_reversal_lambda=0.5),
    # DANNVariant("grl-2p0", gradient_reversal_lambda=2.0),
    # DANNVariant("domain-weight-0p5", domain_loss_weight=0.5),
    # DANNVariant("domain-weight-2p0", domain_loss_weight=2.0),
    # DANNVariant("freeze-backbone", freeze_backbone=True),
]

HELD_OUT_DOMAINS = [
    "Deepfakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
]

def run_experiment(base_config: ConfigModel):

    for held_out_domain in HELD_OUT_DOMAINS:
        for architecture in MODEL_ARCHITECTURES:
            for protocol in PROTOCOLS:
                for seed in SEEDS:
                    baseline_config = _make_config(
                        base_config,
                        mode=ExperimentMode.BASELINE,
                        protocol=protocol,
                        architecture=architecture,
                        held_out_domain=held_out_domain,
                        seed=seed,
                        variant_name="baseline",
                    )
                    orchestrate(baseline_config)

                    for variant in DANN_ABLATIONS:
                        dann_config = _make_config(
                            base_config,
                            mode=ExperimentMode.DANN,
                            protocol=protocol,
                            architecture=architecture,
                            held_out_domain=held_out_domain,
                            seed=seed,
                            variant_name=variant.name,
                            variant=variant,
                        )
                        orchestrate(dann_config)

def _make_config(
    config: ConfigModel,
    mode: ExperimentMode,
    protocol: ExperimentConfig.Protocol,
    architecture: ModelConfig.Architecture,
    held_out_domain: str,
    seed: int,
    variant_name: str,
    variant: DANNVariant | None = None,
) -> ConfigModel:

    experiment_name = _experiment_name(
        config.experiment.name,
        mode.value,
        held_out_domain,
        seed,
        architecture.value,
        protocol.value,
        variant_name,
    )
    domain_adaptation = config.domain_adaptation
    model = replace(config.model, architecture=architecture)

    if variant is not None:

        if variant.gradient_reversal_lambda is not None:
            domain_adaptation = replace(
                domain_adaptation,
                gradient_reversal_lambda=variant.gradient_reversal_lambda,
            )

        if variant.domain_loss_weight is not None:
            domain_adaptation = replace(
                domain_adaptation,
                domain_loss_weight=variant.domain_loss_weight,
            )

        if variant.freeze_backbone is not None:
            model = replace(
                model,
                freeze_backbone=variant.freeze_backbone,
            )

    return replace(
        config,
        experiment=replace(
            config.experiment,
            name=experiment_name,
            mode=mode,
            protocol=protocol,
            held_out_domain=held_out_domain,
        ),
        runtime=replace(config.runtime, seed=seed),
        model=model,
        domain_adaptation=domain_adaptation,
    )

def _experiment_name(
    base_name: str,
    mode: str,
    held_out_domain: str,
    seed: int,
    architecture: str,
    protocol: str,
    variant_name: str,
) -> str:

    return "_".join([
        _safe(base_name),
        _safe(mode),
        _safe(protocol),
        _safe(architecture),
        f"held-out-{_safe(held_out_domain)}",
        f"seed-{seed}",
        _safe(variant_name),
    ])

def _safe(value: str) -> str:
    return (
        value.replace(" ", "-")
        .replace("/", "-")
        .replace(".", "p")
    )

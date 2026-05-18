from dataclasses import dataclass, replace
from src.loaders.config import ConfigModel, ExperimentMode
from src.orchestrator import orchestrate

@dataclass(frozen=True)
class DANNVariant:
    name: str
    gradient_reversal_lambda: float | None = None
    domain_loss_weight: float | None = None
    freeze_backbone: bool | None = None

SEEDS = [42, 123, 2026]

DANN_ABLATIONS = [
    DANNVariant("dann-main"),
    DANNVariant("grl-0p1", gradient_reversal_lambda=0.1),
    DANNVariant("grl-0p5", gradient_reversal_lambda=0.5),
    DANNVariant("grl-2p0", gradient_reversal_lambda=2.0),
    DANNVariant("domain-weight-0p5", domain_loss_weight=0.5),
    DANNVariant("domain-weight-2p0", domain_loss_weight=2.0),
    DANNVariant("freeze-backbone", freeze_backbone=True),
]

def run_experiment(base_config: ConfigModel):

    for held_out_domain in base_config.experiment.fake_domains:
        for seed in SEEDS:
            baseline_config = _make_config(
                base_config,
                mode=ExperimentMode.BASELINE,
                held_out_domain=held_out_domain,
                seed=seed,
                variant_name="baseline",
            )
            orchestrate(baseline_config)
        
            for variant in DANN_ABLATIONS:
                dann_config = _make_config(
                    base_config,
                    mode=ExperimentMode.DANN,
                    held_out_domain=held_out_domain,
                    seed=seed,
                    variant_name=variant.name,
                    variant=variant,
                )
                orchestrate(dann_config)

def _make_config(
    config: ConfigModel,
    mode: ExperimentMode,
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
        variant_name,
    )
    domain_adaptation = config.domain_adaptation
    model = config.model

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
    variant_name: str,
) -> str:

    return "_".join([
        _safe(base_name),
        _safe(mode),
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
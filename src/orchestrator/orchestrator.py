from src.loaders.config import ConfigModel, ExperimentMode
from src.pipelines.data.manifest import ManifestManager, ManifestModel
from src.pipelines.data.split import SplitManager, SplitModel
from src.pipelines.baseline.pipeline import BaselinePipeline
from src.pipelines.dann.pipeline import DANNPipeline

def orchestrate(config: ConfigModel):

    manifest, split = _prepare_data(config)

    match config.experiment.mode:
        case ExperimentMode.BASELINE:
            return baseline(config, split)
        case ExperimentMode.DANN:
            return dann(config, split)
        case _:
            raise ValueError(f"Invalid experiment mode: {config.experiment.mode}")

def baseline(config: ConfigModel, split: SplitModel):
    baseline_pipeline = BaselinePipeline(config, split)
    baseline_pipeline.run()

def dann(config: ConfigModel, split: SplitModel):
    dann_pipeline = DANNPipeline(config, split)
    dann_pipeline.run()

def _prepare_data(config: ConfigModel) -> tuple[ManifestModel, SplitModel]:

    manifest_manager = ManifestManager(
        config.data.dataset_path,
        config.manifest.path,
        config.experiment.real_domain
    )
    manifest: ManifestModel = manifest_manager.make_manifest()

    split_manager = SplitManager(
        manifest,
        config.split,
        config.experiment
    )
    split: SplitModel = split_manager.make_split()

    return manifest, split
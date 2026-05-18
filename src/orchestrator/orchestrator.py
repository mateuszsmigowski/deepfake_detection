from src.loaders.config import ConfigModel, ExperimentMode
from src.pipelines.data.manifest import ManifestManager, ManifestModel
from src.pipelines.data.split import SplitManager, OneOutSplitModel
from src.pipelines.baseline.pipeline import BaselinePipeline
from src.pipelines.dann.pipeline import DANNPipeline

def orchestrate(config: ConfigModel):

    manifest, one_out_split = _prepare_data(config)

    match config.experiment.mode:
        case ExperimentMode.BASELINE:
            return baseline(config, one_out_split)
        case ExperimentMode.DANN:
            return dann(config, one_out_split)
        case _:
            raise ValueError(f"Invalid experiment mode: {config.experiment.mode}")

def baseline(config: ConfigModel, one_out_split: OneOutSplitModel):
    baseline_pipeline = BaselinePipeline(config, one_out_split)
    baseline_pipeline.run()

def dann(config: ConfigModel, one_out_split: OneOutSplitModel):
    dann_pipeline = DANNPipeline(config, one_out_split)
    dann_pipeline.run()

def _prepare_data(config: ConfigModel) -> tuple[ManifestModel, OneOutSplitModel]:

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
    one_out_split: OneOutSplitModel = split_manager.make_split()

    return manifest, one_out_split
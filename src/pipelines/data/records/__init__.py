from src.loaders.config import ConfigModel
from src.pipelines.data.split import SplitModel
from .baseline_preparation import BaselineRecordsPreparation
from .dann_adaptation_preparation import DannAdaptationRecordsPreparation
from .dann_generalization_preparation import DannGeneralizationRecordsPreparation

def prepare_baseline_records(config: ConfigModel, split: SplitModel):
    return BaselineRecordsPreparation(config, split).prepare()

def prepare_dann_generalization_records(
    config: ConfigModel,
    split: SplitModel,
):
    return DannGeneralizationRecordsPreparation(config, split).prepare()

def prepare_dann_adaptation_records(config: ConfigModel, split: SplitModel):
    return DannAdaptationRecordsPreparation(config, split).prepare()

__all__ = [
    "prepare_baseline_records",
    "prepare_dann_generalization_records",
    "prepare_dann_adaptation_records",
]

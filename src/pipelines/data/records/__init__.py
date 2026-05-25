from src.loaders.config import ConfigModel
from src.pipelines.data.split import OneOutSplitModel
from .baseline_preparation import BaselineRecordsPreparation
from .dann_adaptation_preparation import (
    DannAdaptationRecordsPreparation,
    prepare_dann_adaptation_records,
)
from .dann_generalization_preparation import DannGeneralizationRecordsPreparation
from .preparation import RecordsPreparation

def prepare_baseline_records(config: ConfigModel, one_out_split: OneOutSplitModel):
    return BaselineRecordsPreparation(config, one_out_split).prepare()

def prepare_dann_generalization_records(
    config: ConfigModel,
    one_out_split: OneOutSplitModel,
):
    return DannGeneralizationRecordsPreparation(config, one_out_split).prepare()

__all__ = [
    "BaselineRecordsPreparation",
    "DannAdaptationRecordsPreparation",
    "DannGeneralizationRecordsPreparation",
    "RecordsPreparation",
    "prepare_baseline_records",
    "prepare_dann_adaptation_records",
    "prepare_dann_generalization_records",
]

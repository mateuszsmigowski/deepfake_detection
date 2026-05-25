from dataclasses import dataclass
from src.pipelines.data.image_record_model import ImageRecordModel

@dataclass(frozen=True)
class SplitModel:
    train: list[ImageRecordModel]
    validation: list[ImageRecordModel]
    test: list[ImageRecordModel]

@dataclass(frozen=True)
class OneOutSplitModel:
    source: SplitModel
    target: SplitModel
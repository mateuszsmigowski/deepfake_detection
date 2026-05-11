from dataclasses import dataclass
from src.pipelines.data.image_record_model import ImageRecordModel


@dataclass(frozen=True)
class ManifestModel:
    records: list[ImageRecordModel]
    
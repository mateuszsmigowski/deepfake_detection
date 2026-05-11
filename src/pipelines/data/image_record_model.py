from enum import StrEnum
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class ImageRecordModel:

    class Label(StrEnum):
        REAL = "real"
        FAKE = "fake"

    relative_path: Path
    filename: str
    domain: str
    label: Label
    video_id: str
    frame_id: int | None
    group_id: str
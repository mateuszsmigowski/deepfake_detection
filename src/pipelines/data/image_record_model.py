from enum import StrEnum
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class ImageRecordModel:

    class Domain(StrEnum):
        DEEPFAKE = "deepfake"
        ORIGINAL = "original"
        FACE2FACE = "face2face"
        FACESWAP = "faceswap"
        NEURALTEXTURES = "neuraltextures"
        FACESHIFTER = "faceshifter"

    class Label(StrEnum):
        REAL = "real"
        FAKE = "fake"

    relative_path: Path
    filename: str
    domain: Domain
    label: Label
    video_id: str
    frame_id: int | None
    group_id: str
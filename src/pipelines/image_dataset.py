import torch
from typing import Callable
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from src.pipelines.data.image_record_model import ImageRecordModel

class ImageDataset(Dataset):

    label_to_int = {
        ImageRecordModel.Label.REAL: 0,
        ImageRecordModel.Label.FAKE: 1,
    }

    domain_to_int = {
        "Original": 0,
        "Deepfakes": 1,
        "Face2Face": 2,
        "FaceSwap": 3,
        "NeuralTextures": 4,
        "FaceShifter": 5,
    }

    def __init__(
        self,
        records: list[ImageRecordModel],
        dataset_path: Path,
        transforms: Callable | None = None,
    ):
        
        self.records = records
        self.dataset_path = dataset_path
        self.transforms = transforms

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):

        record = self.records[index]
        image = read_image(
            self.dataset_path / record.relative_path,
            ImageReadMode.RGB,
        )

        if self.transforms is not None:
            image = self.transforms(image)

        
        label = torch.tensor(self.label_to_int[record.label], dtype=torch.float32)
        domain = torch.tensor(self.domain_to_int[record.domain], dtype=torch.long)

        metadata = {
            "filename": record.filename,
            "domain": record.domain,
            "label": record.label,
            "video_id": record.video_id,
            "frame_id": record.frame_id,
            "group_id": record.group_id,
        }
        return {
            "image": image,
            "label": label,
            "domain": domain,
            "metadata": metadata,
        }
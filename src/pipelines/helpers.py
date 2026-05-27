import random
from pathlib import Path
from torch.utils.data import DataLoader
from src.pipelines.data.image_record_model import ImageRecordModel
from src.pipelines.image_dataset import ImageDataset
from src.loaders.config import ConfigModel
from src.pipelines.data.split import SplitModel
from src.models.builder import ModelBuilder

def prepare_data_loader(
    config: ConfigModel,
    records: list[ImageRecordModel],
    shuffle: bool = True,
    domain_to_int: dict[str, int] | None = None,
    isTraining: bool = False,
    batch_size: int | None = None,
    drop_last: bool = False,
) -> DataLoader:

    if not records:
        raise ValueError("Cannot create a DataLoader for an empty record list.")

    if isTraining:
        transforms = ModelBuilder.get_training_transforms(config.model.architecture)
    else:
        transforms = ModelBuilder.get_evaluation_transforms(config.model.architecture)

    image_dataset = ImageDataset(
        records=records,
        dataset_path=config.data.dataset_path,
        domain_to_int=(
            domain_to_int 
            if domain_to_int is not None 
            else config.experiment.domain_to_int
        ),
        transforms=transforms,
    )
    return DataLoader(
        image_dataset,
        batch_size=batch_size or config.training.batch_size,
        shuffle=shuffle,
        num_workers=config.runtime.num_workers,
        persistent_workers=config.runtime.num_workers > 0,
        drop_last=drop_last,
    )

def prepare_records(config: ConfigModel, split: SplitModel) -> tuple[
    list[ImageRecordModel],
    list[ImageRecordModel],
    list[ImageRecordModel],
]:

    train_records = _balanced_records(
        _filter_existing_records(split.train, config.data.dataset_path),
        real_limit=config.data.train_source_real,
        fake_limit=config.data.train_source_fake,
    )
    validation_records = _balanced_records(
        _filter_existing_records(split.validation, config.data.dataset_path),
        real_limit=config.data.val_source_real,
        fake_limit=config.data.val_source_fake,
    )
    test_records = _filter_existing_records(split.test, config.data.dataset_path)

    random.shuffle(train_records)
    random.shuffle(validation_records)
    random.shuffle(test_records)

    return train_records, validation_records, test_records

def _filter_existing_records(
    records: list[ImageRecordModel],
    dataset_path: Path,
) -> list[ImageRecordModel]:
    return [
        record
        for record in records
        if (dataset_path / record.relative_path).exists()
    ]

def _balanced_records(
    records: list[ImageRecordModel],
    real_limit: int,
    fake_limit: int,
) -> list[ImageRecordModel]:

    real_records = [
        record for record in records if record.label == ImageRecordModel.Label.REAL
    ]
    fake_records = [
        record for record in records if record.label == ImageRecordModel.Label.FAKE
    ]

    random.shuffle(real_records)
    random.shuffle(fake_records)

    if len(real_records) < real_limit:
        raise ValueError("Not enough real records to satisfy the requested limit.")
    if len(fake_records) < fake_limit:
        raise ValueError("Not enough fake records to satisfy the requested limit.")

    return real_records[:real_limit] + fake_records[:fake_limit]

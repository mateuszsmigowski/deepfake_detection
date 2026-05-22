import random
from pathlib import Path
from torch.utils.data import DataLoader
from src.pipelines.data.image_record_model import ImageRecordModel
from src.pipelines.image_dataset import ImageDataset
from src.loaders.config import ConfigModel, ExperimentConfig
from src.pipelines.data.split import OneOutSplitModel
from src.models.builder import ModelBuilder

def prepare_data_loader(
    config: ConfigModel,
    records: list[ImageRecordModel],
    shuffle: bool = True,
    domain_to_int: dict[str, int] | None = None,
    isTraining: bool = False,
) -> DataLoader:

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
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        num_workers=config.runtime.num_workers,
        persistent_workers=config.runtime.num_workers > 0,
    )

def prepare_records(config: ConfigModel, one_out_split: OneOutSplitModel) -> tuple[
    list[ImageRecordModel],
    list[ImageRecordModel],
    list[ImageRecordModel],
    list[ImageRecordModel],
]:

    source_train = _balanced_records(
        _filter_existing_records(one_out_split.source.train, config.data.dataset_path),
        real_limit=config.data.train_source_real,
        fake_limit=config.data.train_source_fake,
    )
    source_validation = _balanced_records(
        _filter_existing_records(one_out_split.source.validation, config.data.dataset_path),
        real_limit=config.data.val_source_real,
        fake_limit=config.data.val_source_fake,
    )
    target_test = _filter_existing_records(one_out_split.target.test, config.data.dataset_path)

    match config.experiment.protocol:
        case ExperimentConfig.Protocol.GENERALIZATION:
            target_train = []
        case ExperimentConfig.Protocol.ADAPTATION:
            target_train = _limit_fake(
                _filter_existing_records(one_out_split.target.train, config.data.dataset_path),
                config.data.train_target_fake
            )

    random.shuffle(source_train)
    random.shuffle(target_train)
    random.shuffle(source_validation)

    return source_train, target_train, source_validation, target_test

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

    real_records = [record for record in records if record.label == "real"]
    fake_records = [record for record in records if record.label == "fake"]

    random.shuffle(real_records)
    random.shuffle(fake_records)

    return real_records[:real_limit] + fake_records[:fake_limit]

def _limit_fake(records: list[ImageRecordModel], limit: int) -> list[ImageRecordModel]:

    fake_records = [record for record in records if record.label == ImageRecordModel.Label.FAKE]
    random.shuffle(fake_records)
    return fake_records[:limit]
import random
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
from src.pipelines.data.image_record_model import ImageRecordModel
from src.pipelines.image_dataset import ImageDataset
from src.loaders.config import ConfigModel, DataConfig
from src.pipelines.data.split import OneOutSplitModel

def prepare_data_loader(
    config: ConfigModel,
    records: list[ImageRecordModel],
    shuffle: bool = True,
    domain_to_int: dict[str, int] | None = None,
) -> DataLoader:

    # TODO: Transofrms shouldn't be hardcoded here
    weights = ResNet18_Weights.DEFAULT
    transforms = weights.transforms()

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

def prepare_records(data_config: DataConfig, one_out_split: OneOutSplitModel) -> tuple[
    list[ImageRecordModel],
    list[ImageRecordModel],
    list[ImageRecordModel],
]:
    
    train_records = _balanced_records(
        one_out_split.source.train,
        real_limit=data_config.train_real,
        fake_limit=data_config.train_fake,
    )
    val_records = _balanced_records(
        one_out_split.source.validation,
        real_limit=data_config.val_real,
        fake_limit=data_config.val_fake,
    )
    test_records = list(one_out_split.target.test)

    random.shuffle(train_records)
    random.shuffle(val_records)

    return train_records, val_records, test_records

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
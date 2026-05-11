import random
from src.loaders.config import ConfigModel
from src.pipelines.data.split import OneOutSplitModel
from src.pipelines.data.image_record_model import ImageRecordModel
from torch.utils.data import DataLoader
from src.models.dann.model import DANNClassifier
from src.training.dann.trainer import DANNTrainer
from src.pipelines.image_dataset import ImageDataset
from torchvision.models import ResNet18_Weights

class DANNPipeline:

    def __init__(self, config: ConfigModel, one_out_split: OneOutSplitModel):
        self.config = config
        self.one_out_split = one_out_split

    def run(self):

        train_records, val_records, test_records = self._prepare_records()

        train_loader = self._prepare_data_loader(train_records, shuffle=True)
        val_loader = self._prepare_data_loader(val_records, shuffle=False)
        test_loader = self._prepare_data_loader(test_records, shuffle=False)
        classifier = DANNClassifier(self.config.model, self.config.domain_adaptation)
        trainer = DANNTrainer(self.config, classifier, train_loader, val_loader, test_loader)
        trainer.run()

    def _prepare_data_loader(self,
        records: list[ImageRecordModel],
        shuffle: bool = True
    ) -> DataLoader:

        # TODO: Transofrms shouldn't be hardcoded here
        weights = ResNet18_Weights.DEFAULT
        transforms = weights.transforms()

        image_dataset = ImageDataset(
            records=records,
            dataset_path=self.config.data.dataset_path,
            transforms=transforms,
        )
        return DataLoader(
            image_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=shuffle,
            num_workers=self.config.runtime.num_workers,
            persistent_workers=self.config.runtime.num_workers > 0,
        )

    def _prepare_records(self) -> tuple[
        list[ImageRecordModel],
        list[ImageRecordModel],
        list[ImageRecordModel],
    ]:
        
        train_records = self._balanced_records(
            self.one_out_split.source.train,
            real_limit=3500,
            fake_limit=3500,
        )
        val_records = self._balanced_records(
            self.one_out_split.source.validation,
            real_limit=500,
            fake_limit=500,
        )
        test_records = list(self.one_out_split.target.test)

        random.shuffle(train_records)
        random.shuffle(val_records)

        return train_records, val_records, test_records

    def _balanced_records(
        self,
        records: list[ImageRecordModel],
        real_limit: int,
        fake_limit: int,
    ) -> list[ImageRecordModel]:

        real_records = [record for record in records if record.label == "real"]
        fake_records = [record for record in records if record.label == "fake"]

        random.shuffle(real_records)
        random.shuffle(fake_records)

        return real_records[:real_limit] + fake_records[:fake_limit]
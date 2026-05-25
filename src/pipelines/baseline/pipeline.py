from src.pipelines.data.split import OneOutSplitModel
from src.loaders.config import ConfigModel
from src.models.baseline.model import BaselineClassifier
from src.training import configure_reproducibility, BaselineTrainer
from src.pipelines.helpers import prepare_data_loader
from src.pipelines.data.records import BaselineRecordsPreparation
from src.pipelines.data.image_record_model import ImageRecordModel
from torch.utils.data import DataLoader


class BaselinePipeline:

    def __init__(self, config: ConfigModel, one_out_split: OneOutSplitModel):

        self.config = config
        self.one_out_split = one_out_split

    def run(self):

        configure_reproducibility(self.config.runtime.seed, self.config.runtime.deterministic)

        train_records, val_records, test_records = self._prepare_records()
        train_loader, val_loader, test_loader = self._prepare_data_loaders(
            train_records,
            val_records,
            test_records,
        )
        classifier = BaselineClassifier(self.config.model)
        trainer = self._prepare_trainer(classifier, train_loader, val_loader, test_loader)
        trainer.run()

    def _prepare_records(self):
        return BaselineRecordsPreparation(self.config, self.one_out_split).prepare()

    def _prepare_data_loaders(
        self,
        train_records: list[ImageRecordModel],
        val_records: list[ImageRecordModel],
        test_records: list[ImageRecordModel],
    ):
        train_loader = prepare_data_loader(self.config, train_records, shuffle=True, isTraining=True)
        val_loader = prepare_data_loader(self.config, val_records, shuffle=False)
        test_loader = prepare_data_loader(self.config, test_records, shuffle=False)
        return train_loader, val_loader, test_loader

    def _prepare_trainer(
        self,
        classifier: BaselineClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
    ):
        return BaselineTrainer(self.config, classifier, train_loader, val_loader, test_loader)
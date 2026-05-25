from src.loaders.config import ConfigModel, ExperimentConfig
from src.pipelines.data.split import OneOutSplitModel
from src.pipelines.data.image_record_model import ImageRecordModel
from torch.utils.data import DataLoader
from src.models.dann.model import DANNClassifier
from src.training import configure_reproducibility, DANNGeneralizationTrainer, DANNAdaptationTrainer
from src.pipelines.data.records import (
    DannGeneralizationRecordsPreparation,
    DannAdaptationRecordsPreparation,
)
from src.pipelines.helpers import prepare_data_loader

class DANNPipeline:

    def __init__(self, config: ConfigModel, one_out_split: OneOutSplitModel):
        self.config = config
        self.one_out_split = one_out_split

    def run(self):

        configure_reproducibility(
            self.config.runtime.seed,
            self.config.runtime.deterministic,
        )
        train_records, target_train, val_records, test_records = self._prepare_records()
        source_train_loader, target_train_loader, val_loader, test_loader = self._prepare_data_loaders(
            train_records,
            target_train,
            val_records,
            test_records,
        )
        classifier = DANNClassifier(
            self.config.model,
            self.config.domain_adaptation,
            domains_count=self._get_domains_count(),
        )
        trainer = self._prepare_trainer(
            classifier,
            source_train_loader,
            target_train_loader,
            val_loader,
            test_loader,
        )

        trainer.run()

    def _prepare_records(self):

        match self.config.experiment.protocol:
            case ExperimentConfig.Protocol.GENERALIZATION:
                records_preparation = DannGeneralizationRecordsPreparation(
                    self.config,
                    self.one_out_split,
                )
                return records_preparation.prepare()
            case ExperimentConfig.Protocol.ADAPTATION:
                records_preparation = DannAdaptationRecordsPreparation(
                    self.config,
                    self.one_out_split,
                )
                return records_preparation.prepare()

    def _prepare_data_loaders(
        self,
        train_records: list[ImageRecordModel],
        target_train: list[ImageRecordModel],
        val_records: list[ImageRecordModel],
        test_records: list[ImageRecordModel],
    ) -> tuple[DataLoader, DataLoader | None, DataLoader, DataLoader]:

        match self.config.experiment.protocol:
            case ExperimentConfig.Protocol.GENERALIZATION:
                return self._prepare_generalization_data_loaders(
                    train_records,
                    val_records,
                    test_records,
                )
            case ExperimentConfig.Protocol.ADAPTATION:
                return self._prepare_adaptation_data_loaders(
                    train_records,
                    target_train,
                    val_records,
                    test_records,
                )
            case _:
                raise ValueError(f"Invalid protocol: {self.config.experiment.protocol}")

    def _prepare_generalization_data_loaders(
        self,
        train_records: list[ImageRecordModel],
        val_records: list[ImageRecordModel],
        test_records: list[ImageRecordModel],
    ) -> tuple[DataLoader, None, DataLoader, DataLoader]:

        source_domain_to_int = self.config.experiment.source_domain_to_int

        source_train_loader = prepare_data_loader(
            self.config,
            train_records,
            shuffle=True,
            domain_to_int=source_domain_to_int,
            isTraining=True,
        )
        val_loader = prepare_data_loader(
            self.config,
            val_records,
            shuffle=False,
            domain_to_int=source_domain_to_int,
        )
        test_loader = prepare_data_loader(
            self.config,
            test_records,
            shuffle=False,
        )
        return source_train_loader, None, val_loader, test_loader

    def _prepare_adaptation_data_loaders(
        self,
        train_records: list[ImageRecordModel],
        target_train: list[ImageRecordModel],
        val_records: list[ImageRecordModel],
        test_records: list[ImageRecordModel],
    ) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:

        source_domain_to_int = self.config.experiment.source_domain_to_int
        target_domain_to_int = {self.config.experiment.held_out_domain: 1}

        source_train_loader = prepare_data_loader(
            self.config,
            train_records,
            shuffle=True,
            domain_to_int=source_domain_to_int,
            isTraining=True,
        )
        target_train_loader = prepare_data_loader(
            self.config,
            target_train,
            shuffle=True,
            domain_to_int=target_domain_to_int,
            isTraining=True,
        )
        val_loader = prepare_data_loader(
            self.config,
            val_records,
            shuffle=False,
            domain_to_int=source_domain_to_int,
        )
        test_loader = prepare_data_loader(
            self.config,
            test_records,
            shuffle=False,
        )
        return source_train_loader, target_train_loader, val_loader, test_loader

    def _prepare_trainer(
        self,
        classifier: DANNClassifier,
        source_train_loader: DataLoader,
        target_train_loader: DataLoader | None,
        val_loader: DataLoader,
        test_loader: DataLoader,
    ):

        match self.config.experiment.protocol:
            case ExperimentConfig.Protocol.GENERALIZATION:
                trainer = DANNGeneralizationTrainer(
                    self.config,
                    classifier,
                    source_train_loader,
                    val_loader,
                    test_loader,
                )
            case ExperimentConfig.Protocol.ADAPTATION:
                if target_train_loader is None:
                    raise ValueError("Target train loader is required for DANN adaptation.")

                trainer = DANNAdaptationTrainer(
                    self.config,
                    classifier,
                    source_train_loader,
                    target_train_loader,
                    val_loader,
                    test_loader,
                )
        return trainer

    def _get_domains_count(self) -> int:

        match self.config.experiment.protocol:
            case ExperimentConfig.Protocol.GENERALIZATION:
                return len(self.config.experiment.source_domain_to_int)
            case ExperimentConfig.Protocol.ADAPTATION:
                return 2 # TODO: Remove magic number, should be calculated

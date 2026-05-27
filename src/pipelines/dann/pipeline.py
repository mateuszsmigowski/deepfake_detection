from src.loaders.config import ConfigModel, ExperimentConfig
from src.pipelines.data.split import SplitModel
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

    def __init__(self, config: ConfigModel, split: SplitModel):
        self.config = config
        self.split = split

    def run(self):

        configure_reproducibility(
            self.config.runtime.seed,
            self.config.runtime.deterministic,
        )
        train_records, val_records, test_records = self._prepare_records()
        classifier = DANNClassifier(
            self.config.model,
            self.config.domain_adaptation,
            domains_count=self._get_domains_count(),
        )

        match self.config.experiment.protocol:
            case ExperimentConfig.Protocol.GENERALIZATION:
                train_loader, val_loader, test_loader = self._prepare_generalization_data_loaders(
                    train_records,
                    val_records,
                    test_records,
                )
                trainer = DANNGeneralizationTrainer(
                    self.config,
                    classifier,
                    train_loader,
                    val_loader,
                    test_loader,
                )
            case ExperimentConfig.Protocol.ADAPTATION:
                (
                    source_label_loader,
                    source_domain_loader,
                    target_domain_loader,
                    val_loader,
                    test_loader,
                ) = self._prepare_adaptation_data_loaders(
                    train_records,
                    val_records,
                    test_records,
                )
                trainer = DANNAdaptationTrainer(
                    self.config,
                    classifier,
                    source_label_loader,
                    source_domain_loader,
                    target_domain_loader,
                    val_loader,
                    test_loader,
                )
            case _:
                raise ValueError(f"Invalid protocol: {self.config.experiment.protocol}")

        trainer.run()

    def _prepare_records(self):

        match self.config.experiment.protocol:
            case ExperimentConfig.Protocol.GENERALIZATION:
                records_preparation = DannGeneralizationRecordsPreparation(
                    self.config,
                    self.split,
                )
                return records_preparation.prepare()
            case ExperimentConfig.Protocol.ADAPTATION:
                records_preparation = DannAdaptationRecordsPreparation(
                    self.config,
                    self.split,
                )
                return records_preparation.prepare()

    def _prepare_generalization_data_loaders(
        self,
        train_records: list[ImageRecordModel],
        val_records: list[ImageRecordModel],
        test_records: list[ImageRecordModel],
    ) -> tuple[DataLoader, DataLoader, DataLoader]:

        domain_to_int = self._fake_domain_to_int()

        train_loader = prepare_data_loader(
            self.config,
            train_records,
            shuffle=True,
            domain_to_int=domain_to_int,
            isTraining=True,
        )
        val_loader = prepare_data_loader(
            self.config,
            val_records,
            shuffle=False,
            domain_to_int=domain_to_int,
        )
        test_loader = prepare_data_loader(
            self.config,
            test_records,
            shuffle=False,
        )
        return train_loader, val_loader, test_loader

    def _prepare_adaptation_data_loaders(
        self,
        train_records: list[ImageRecordModel],
        val_records: list[ImageRecordModel],
        test_records: list[ImageRecordModel],
    ) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader, DataLoader]:

        source_label_records = self._adaptation_source_label_records(train_records)
        source_domain_records = self._adaptation_source_domain_records(train_records)
        target_domain_records = self._adaptation_target_domain_records(train_records)
        domain_batch_size = self._adaptation_domain_batch_size(
            source_domain_records,
            target_domain_records,
        )
        domain_to_int = self._fake_domain_to_int()

        source_label_loader = prepare_data_loader(
            self.config,
            source_label_records,
            shuffle=True,
            domain_to_int=domain_to_int,
            isTraining=True,
        )
        source_domain_loader = prepare_data_loader(
            self.config,
            source_domain_records,
            shuffle=True,
            domain_to_int=domain_to_int,
            isTraining=True,
            batch_size=domain_batch_size,
            drop_last=True,
        )
        target_domain_loader = prepare_data_loader(
            self.config,
            target_domain_records,
            shuffle=True,
            domain_to_int=domain_to_int,
            isTraining=True,
            batch_size=domain_batch_size,
            drop_last=True,
        )
        val_loader = prepare_data_loader(
            self.config,
            val_records,
            shuffle=False,
            domain_to_int=domain_to_int,
        )
        test_loader = prepare_data_loader(
            self.config,
            test_records,
            shuffle=False,
        )
        return (
            source_label_loader,
            source_domain_loader,
            target_domain_loader,
            val_loader,
            test_loader,
        )

    def _adaptation_source_label_records(
        self,
        train_records: list[ImageRecordModel],
    ) -> list[ImageRecordModel]:
        source_domains = set(self.config.experiment.source_domains)
        return [record for record in train_records if record.domain in source_domains]

    def _adaptation_source_domain_records(
        self,
        train_records: list[ImageRecordModel],
    ) -> list[ImageRecordModel]:
        source_domains = set(self.config.experiment.source_domains)
        return [
            record
            for record in train_records
            if (
                record.domain in source_domains
                and record.label == ImageRecordModel.Label.FAKE
            )
        ]

    def _adaptation_target_domain_records(
        self,
        train_records: list[ImageRecordModel],
    ) -> list[ImageRecordModel]:
        held_out_domain = self.config.experiment.held_out_domain
        return [
            record
            for record in train_records
            if (
                record.domain == held_out_domain
                and record.label == ImageRecordModel.Label.FAKE
            )
        ]

    def _adaptation_domain_batch_size(
        self,
        source_domain_records: list[ImageRecordModel],
        target_domain_records: list[ImageRecordModel],
    ) -> int:
        if not source_domain_records:
            raise ValueError("Adaptation requires source fake records for domain loss.")
        if not target_domain_records:
            raise ValueError("Adaptation requires target fake records for domain loss.")

        requested = (
            self.config.domain_adaptation.domain_batch_size
            or max(1, self.config.training.batch_size // 2)
        )
        if requested <= 0:
            raise ValueError("Domain batch size must be greater than zero.")

        return min(requested, len(source_domain_records), len(target_domain_records))

    def _get_domains_count(self) -> int:

        match self.config.experiment.protocol:
            case ExperimentConfig.Protocol.GENERALIZATION:
                return len(self.config.experiment.fake_domains)
            case ExperimentConfig.Protocol.ADAPTATION:
                return len(self.config.experiment.fake_domains)

    def _fake_domain_to_int(self) -> dict[str, int]:
        return self.config.experiment.fake_domain_to_int

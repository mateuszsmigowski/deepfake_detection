import random
from src.loaders.config import ConfigModel
from src.pipelines.data.split import OneOutSplitModel
from src.pipelines.data.image_record_model import ImageRecordModel
from src.pipelines.data.records.preparation import RecordsPreparation

class DannGeneralizationRecordsPreparation(RecordsPreparation):

    def __init__(self, config: ConfigModel, one_out_split: OneOutSplitModel):
        super().__init__(config, one_out_split)

    def prepare(self):

        fake_domains = self._source_fake_domains()

        train_fake_per_domain = self._resolve_per_domain_limit(
            self.config.data.train_source_fake, 
            fake_domains,
        )
        val_fake_per_domain = self._resolve_per_domain_limit(
            self.config.data.val_source_fake, 
            fake_domains,
        )
        train_records = self._balanced_per_domain_records(
            self._filter_existing_records(self.one_out_split.source.train),
            self.config.data.train_source_real,
            fake_domains,
            train_fake_per_domain,
        )
        val_records = self._balanced_per_domain_records(
            self._filter_existing_records(self.one_out_split.source.validation),
            self.config.data.val_source_real,
            fake_domains,
            val_fake_per_domain,
        )
        test_records = self._filter_existing_records(
            self.one_out_split.target.test,
        )
        self._shuffle(train_records, val_records, test_records)

        self._log_split_counts(
            train_records,
            "train",
            tag="dann_generalization",
            extra=f", real: {self.config.data.train_source_real}, fake_per_domain: {train_fake_per_domain}",
        )
        self._log_split_counts(
            val_records,
            "val",
            tag="dann_generalization",
            extra=f", real: {self.config.data.val_source_real}, fake_per_domain: {val_fake_per_domain}",
        )
        self._log_split_counts(test_records, "test", tag="dann_generalization")

        return train_records, [], val_records, test_records

    def _balanced_per_domain_records(
        self,
        records: list[ImageRecordModel],
        real_limit: int,
        fake_domains: list[str],
        fake_per_domain_limit: int,
    ) -> list[ImageRecordModel]:

        real_records = self._limit_real_records(
            records,
            real_limit,
        )
        fake_records = self._limit_fake_records_per_domain(
            records,
            fake_domains,
            fake_per_domain_limit,
        )

        return real_records + fake_records


import random
from pathlib import Path
from src.loaders.config import ConfigModel
from src.pipelines.data.split import SplitModel
from src.pipelines.data.image_record_model import ImageRecordModel
from src.pipelines.data.records.preparation import RecordsPreparation

class BaselineRecordsPreparation(RecordsPreparation):

    def __init__(self, config: ConfigModel, split: SplitModel):
        super().__init__(config, split)

    def prepare(self):

        train_records = self._balanced_baseline_records(
            self._filter_existing_records(self.split.train),
            self.config.data.train_source_real,
            self.config.data.train_source_fake,
        )
        val_records = self._balanced_baseline_records(
            self._filter_existing_records(self.split.validation),
            self.config.data.val_source_real,
            self.config.data.val_source_fake,
        )
        test_records = self._filter_existing_records(
            self.split.test,
        )
        self._shuffle(train_records, val_records, test_records)

        self._log_split_counts(train_records, "train", tag="baseline")
        self._log_split_counts(val_records, "val", tag="baseline")
        self._log_split_counts(test_records, "test", tag="baseline")

        return train_records, val_records, test_records

    def _balanced_baseline_records(
        self,
        records: list[ImageRecordModel],
        real_limit: int,
        fake_limit: int,
    ) -> list[ImageRecordModel]:

        fake_domains = self._source_fake_domains()
        fake_per_domain_limit = self._resolve_per_domain_limit(
            fake_limit,
            fake_domains,
        )

        real_records = self._limit_real_records(records, real_limit)
        fake_records = self._limit_fake_records_per_domain(
            records,
            fake_domains,
            fake_per_domain_limit,
        )
        return real_records + fake_records
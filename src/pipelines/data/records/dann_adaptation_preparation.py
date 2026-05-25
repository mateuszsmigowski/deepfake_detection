import random
from src.loaders.config import ConfigModel
from src.pipelines.data.split import OneOutSplitModel
from src.pipelines.data.image_record_model import ImageRecordModel
from src.pipelines.data.records.preparation import RecordsPreparation

class DannAdaptationRecordsPreparation(RecordsPreparation):

    def __init__(self, config: ConfigModel, one_out_split: OneOutSplitModel):
        super().__init__(config, one_out_split)

    def prepare(self) -> tuple[
        list[ImageRecordModel],
        list[ImageRecordModel],
        list[ImageRecordModel],
        list[ImageRecordModel],
    ]:

        fake_domains = self._source_fake_domains()
        train_per_domain = self._resolve_per_domain_limit(
            self.config.data.train_source_fake,
            fake_domains,
        )
        val_per_domain = self._resolve_per_domain_limit(
            self.config.data.val_source_fake,
            fake_domains,
        )

        train_records = self._balanced_adaptation_source_records(
            self._filter_existing_records(self.one_out_split.source.train),
            fake_domains,
            train_per_domain,
        )
        target_train_records = self._limit_target_fake_records(
            self._filter_existing_records(self.one_out_split.target.train),
            self.config.data.train_target_fake,
        )
        val_records = self._balanced_adaptation_source_records(
            self._filter_existing_records(self.one_out_split.source.validation),
            fake_domains,
            val_per_domain,
        )
        test_records = self._filter_existing_records(self.one_out_split.target.test)

        self._shuffle(train_records, target_train_records, val_records, test_records)

        self._log_split_counts(
            train_records,
            "source_train",
            tag="dann_adaptation",
            extra=f", per_domain: {train_per_domain}",
        )
        self._log_split_counts(target_train_records, "target_train", tag="dann_adaptation")
        self._log_split_counts(
            val_records,
            "val",
            tag="dann_adaptation",
            extra=f", per_domain: {val_per_domain}",
        )
        self._log_split_counts(test_records, "test", tag="dann_adaptation")

        return train_records, target_train_records, val_records, test_records

    def _balanced_adaptation_source_records(
        self,
        records: list[ImageRecordModel],
        fake_domains: list[str],
        per_domain_limit: int,
    ) -> list[ImageRecordModel]:

        real_limit = per_domain_limit * len(fake_domains)
        real_records = self._limit_real_records(records, real_limit)
        fake_records = self._limit_fake_records_per_domain(
            records,
            fake_domains,
            per_domain_limit,
        )

        return real_records + fake_records

    def _limit_target_fake_records(
        self,
        records: list[ImageRecordModel],
        limit: int,
    ) -> list[ImageRecordModel]:

        held_out_domain = self.config.experiment.held_out_domain
        target_fake_records = [
            record
            for record in records
            if record.label == ImageRecordModel.Label.FAKE and record.domain == held_out_domain
        ]
        random.shuffle(target_fake_records)

        if len(target_fake_records) < limit:
            raise ValueError("Not enough target fake records to limit")

        return target_fake_records[:limit]

def prepare_dann_adaptation_records(
    config: ConfigModel,
    one_out_split: OneOutSplitModel,
) -> tuple[
    list[ImageRecordModel],
    list[ImageRecordModel],
    list[ImageRecordModel],
    list[ImageRecordModel],
]:
    return DannAdaptationRecordsPreparation(config, one_out_split).prepare()

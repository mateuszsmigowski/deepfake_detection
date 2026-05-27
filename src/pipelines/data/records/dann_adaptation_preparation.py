import random
from src.loaders.config import ConfigModel
from src.pipelines.data.split import SplitModel
from src.pipelines.data.image_record_model import ImageRecordModel
from src.pipelines.data.records.preparation import RecordsPreparation

class DannAdaptationRecordsPreparation(RecordsPreparation):

    def __init__(self, config: ConfigModel, split: SplitModel):
        super().__init__(config, split)

    def prepare(self) -> tuple[
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

        filtered_train = self._filter_existing_records(self.split.train)
        source_records = self._balanced_adaptation_source_records(
            filtered_train,
            fake_domains,
            train_per_domain,
        )
        target_fake_records = self._limit_target_fake_records(
            filtered_train,
            self.config.data.train_target_fake,
        )
        train_records = source_records + target_fake_records
        val_records = self._balanced_adaptation_source_records(
            self._filter_existing_records(self.split.validation),
            fake_domains,
            val_per_domain,
        )
        test_records = self._filter_existing_records(self.split.test)

        self._shuffle(train_records, val_records, test_records)

        self._log_split_counts(
            train_records,
            "train",
            tag="dann_adaptation",
            extra=f", per_domain: {train_per_domain}, target_fake: {self.config.data.train_target_fake}",
        )
        self._log_split_counts(
            val_records,
            "val",
            tag="dann_adaptation",
            extra=f", per_domain: {val_per_domain}",
        )
        self._log_split_counts(test_records, "test", tag="dann_adaptation")

        return train_records, val_records, test_records

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
    split: SplitModel,
) -> tuple[
    list[ImageRecordModel],
    list[ImageRecordModel],
    list[ImageRecordModel],
]:
    return DannAdaptationRecordsPreparation(config, split).prepare()

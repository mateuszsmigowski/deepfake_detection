import random
from pathlib import Path
from src.pipelines.data.image_record_model import ImageRecordModel
from src.loaders.config import ConfigModel
from src.pipelines.data.split import SplitModel

class RecordsPreparation:

    def __init__(self, config: ConfigModel, split: SplitModel):
        self.config = config
        self.split = split

    def prepare(self):
        raise NotImplementedError("Subclasses must implement this method")

    def _filter_existing_records(self, records: list[ImageRecordModel]) -> list[ImageRecordModel]:

        dataset_path = Path(self.config.data.dataset_path)
        return [
            record
            for record in records
            if (dataset_path / record.relative_path).exists()
        ]

    def _source_fake_domains(self) -> list[str]:
        return [
            domain
            for domain in self.config.experiment.fake_domains
            if domain != self.config.experiment.held_out_domain
        ]

    def _resolve_per_domain_limit(
        self,
        total_fake_limit: int,
        fake_domains: list[str],
        ) -> int:
        
        if not fake_domains:
            raise ValueError("Fake domains are empty")
        if total_fake_limit % len(fake_domains) != 0:
            raise ValueError("Total fake limit is not divisible by the number of fake domains")

        return total_fake_limit // len(fake_domains)

    def _limit_real_records(
        self,
        records: list[ImageRecordModel],
        limit: int,
    ) -> list[ImageRecordModel]:

        real_domain = self.config.experiment.real_domain
        real_records = [
            record
            for record in records
            if record.label == ImageRecordModel.Label.REAL and record.domain == real_domain
        ]
        random.shuffle(real_records)

        if len(real_records) < limit:
            raise ValueError("Not enough real records to limit")

        return real_records[:limit]

    def _limit_fake_records_per_domain(
        self,
        records: list[ImageRecordModel],
        fake_domains: list[str],
        per_domain_limit: int,
    ) -> list[ImageRecordModel]:

        selected_records: list[ImageRecordModel] = []

        for domain in fake_domains:
            domain_fake_records = [
                record
                for record in records
                if record.label == ImageRecordModel.Label.FAKE and record.domain == domain
            ]
            random.shuffle(domain_fake_records)

            if len(domain_fake_records) < per_domain_limit:
                raise ValueError("Not enough fake records to limit")

            selected_records.extend(domain_fake_records[:per_domain_limit])

        return selected_records

    def _shuffle(self, *records: list[ImageRecordModel]) -> None:
        for record in records:
            random.shuffle(record)

    def _log_split_counts(
        self, 
        records: list[ImageRecordModel], 
        split_name: str, 
        tag: str = "", 
        extra: str = "",
    ) -> None:
        counts = {}
        for record in records:
            counts.setdefault(record.domain, {"real": 0, "fake": 0})
            counts[record.domain][record.label.value] += 1
        print(f"[{tag}: {split_name}{extra}] {counts}")
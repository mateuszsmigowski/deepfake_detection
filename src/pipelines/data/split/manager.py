import csv
from pathlib import Path
from src.pipelines.data.manifest import ManifestModel
from src.pipelines.data.split.split_model import SplitModel
from src.pipelines.data.image_record_model import ImageRecordModel
from src.loaders.config import SplitConfig, ExperimentConfig
from src.pipelines.data.split.builder import SplitBuilder


_MANIFEST_FIELDNAMES = [
    "relative_path",
    "filename",
    "domain",
    "label",
    "video_id",
    "frame_id",
    "group_id",
]

class SplitManager:

    @property
    def _split_dir(self) -> Path:
        split_signature = (
            f"seed-{self.split_config.seed}"
            f"_train-{self.split_config.train_ratio:g}"
            f"_val-{self.split_config.validation_ratio:g}"
            f"_test-{self.split_config.test_ratio:g}"
        )
        return (
            self.split_config.path
            / f"held_out_{self.experiment_config.held_out_domain}"
            / self.experiment_config.protocol.value
            / split_signature
        )

    # MARK: - Initialization

    def __init__(self, manifest: ManifestModel, split_config: SplitConfig, experiment_config: ExperimentConfig):

        self.manifest = manifest
        self.split_config = split_config
        self.experiment_config = experiment_config
        self.split_builder = SplitBuilder(manifest, split_config, experiment_config)

    def make_split(self) -> SplitModel:

        if self._split_exists():
            return self._read_split()
        else:
            split = self.split_builder.build_split()
            self._write_split(split)
            return split

    def _split_exists(self) -> bool:
        return all(path.exists() for path in self._split_paths())

    def _write_split(self, split: SplitModel) -> None:

        self._split_dir.mkdir(parents=True, exist_ok=True)

        train_path, validation_path, test_path = self._split_paths()

        self._write_split_csv(train_path, split.train)
        self._write_split_csv(validation_path, split.validation)
        self._write_split_csv(test_path, split.test)

    def _write_split_csv(self, path: Path, records: list[ImageRecordModel]) -> None:

        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=_MANIFEST_FIELDNAMES)
            writer.writeheader()

            for record in records:  
                writer.writerow({
                    "relative_path": record.relative_path.as_posix(),
                    "filename": record.filename,
                    "domain": record.domain,
                    "label": record.label,
                    "video_id": record.video_id,
                    "frame_id": record.frame_id,
                    "group_id": record.group_id,
                })

    def _read_split(self) -> SplitModel:

        train_path, validation_path, test_path = self._split_paths()

        train_records = self._read_split_csv(train_path)
        validation_records = self._read_split_csv(validation_path)
        test_records = self._read_split_csv(test_path)

        return SplitModel(train=train_records, validation=validation_records, test=test_records)

    def _read_split_csv(self, path: Path) -> list[ImageRecordModel]:

        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            return [ImageRecordModel(
                relative_path=Path(record["relative_path"]),
                filename=record["filename"],
                domain=record["domain"],
                label=ImageRecordModel.Label(record["label"]),
                video_id=record["video_id"],
                frame_id=int(record["frame_id"]) if record["frame_id"] else None,
                group_id=record["group_id"],
            ) for record in reader]

    def _split_paths(self) -> tuple[Path, Path, Path]:
        return (
            self._split_dir / "train.csv",
            self._split_dir / "validation.csv",
            self._split_dir / "test.csv",
        )

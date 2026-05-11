from pathlib import Path
import csv
from src.pipelines.data.manifest.manifest_model import ManifestModel
from src.pipelines.data.manifest.parser import ImageDirectoryParser
from src.pipelines.data.image_record_model import ImageRecordModel

# MARK: - Constants
# TODO: Find better way to determine real domain
_MANIFEST_FIELDNAMES = [
    "relative_path",
    "filename",
    "domain",
    "label",
    "video_id",
    "frame_id",
    "group_id",
]

class ManifestManager:

    # MARK: - Initialization

    def __init__(self, dataset_path: Path, output_path: Path):

        self.dataset_path = dataset_path
        self.output_path = output_path
        self.parser = ImageDirectoryParser(dataset_path)

    # MARK: - Public methods

    def make_manifest(self) -> ManifestModel:
        if self.output_path.exists():
            return self._read_manifest()
        else:
            records = self.parser.parse()
            manifest = ManifestModel(records=records)
            self._write_manifest(manifest)
            return manifest

    # MARK: - Private methods

    def _write_manifest(self, manifest: ManifestModel) -> None:

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with self.output_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=_MANIFEST_FIELDNAMES)
            writer.writeheader()

            for record in manifest.records:
                writer.writerow({
                    "relative_path": record.relative_path.as_posix(),
                    "filename": record.filename,
                    "domain": record.domain,
                    "label": record.label.value,
                    "video_id": record.video_id,
                    "frame_id": record.frame_id,
                    "group_id": record.group_id,
                })

    def _read_manifest(self) -> ManifestModel:

        with self.output_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            return ManifestModel(records=[ImageRecordModel(
                relative_path=Path(record["relative_path"]),
                filename=record["filename"],
                domain=record["domain"],
                label=record["label"],
                video_id=record["video_id"],
                frame_id=record["frame_id"],
                group_id=record["group_id"],
            ) for record in reader])
 
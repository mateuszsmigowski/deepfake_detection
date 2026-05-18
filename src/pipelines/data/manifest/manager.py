from pathlib import Path
import csv
from src.pipelines.data.manifest.manifest_model import ManifestModel
from src.pipelines.data.manifest.parser import ImageDirectoryParser
from src.pipelines.data.image_record_model import ImageRecordModel

# MARK: - Constants
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

    def __init__(self, dataset_path: Path, output_path: Path, real_domain: str):

        self.dataset_path = dataset_path
        self.output_path = output_path
        self.real_domain = real_domain
        self.parser = ImageDirectoryParser(dataset_path, real_domain)

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
                label=(
                    ImageRecordModel.Label.REAL
                    if record["domain"] == self.real_domain
                    else ImageRecordModel.Label.FAKE
                ),
                video_id=record["video_id"],
                frame_id=(
                    int(record["frame_id"]) 
                    if record["frame_id"] 
                    else None
                ),
                group_id=record["group_id"],
            ) for record in reader])
 
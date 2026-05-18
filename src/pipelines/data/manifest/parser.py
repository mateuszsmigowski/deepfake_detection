from pathlib import Path
from src.pipelines.data.image_record_model import ImageRecordModel

ImageLabel = ImageRecordModel.Label

# MARK: - Constants
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

class ImageDirectoryParser:

    def __init__(self, dataset_path: Path, real_domain: str):
        self.dataset_path = dataset_path
        self.real_domain = real_domain

    def parse(self) -> list[ImageRecordModel]:
        
        self._validate_dataset_path()
        return self._parse_domains()

    # MARK: - Private methods

    def _parse_domains(self) -> list[ImageRecordModel]:

        domain_paths: list[Path] = self._domain_paths()
        
        image_records: list[ImageRecordModel] = []
        for domain_path in domain_paths:
            image_records.extend(self._parse_images(domain_path))
        return image_records

    def _parse_images(self, domain_path: Path) -> list[ImageRecordModel]:

        domain: str = domain_path.name
        label: ImageLabel = ImageLabel.REAL if domain == self.real_domain else ImageLabel.FAKE
        
        image_paths: list[Path] = self._image_paths(domain_path)
        image_records: list[ImageRecordModel] = []

        for image_path in image_paths:
            video_id, frame_id = self._identify_image(image_path)
            image_records.append(ImageRecordModel(
                relative_path=image_path.relative_to(self.dataset_path),
                filename=image_path.name,
                domain=domain,
                label=label,
                video_id=video_id,
                frame_id=frame_id,
                group_id=f"{domain}:{video_id}",
            ))
        return image_records

    def _identify_image(self, image_path: Path) -> tuple[str, int | None]:
        stem = image_path.stem
        video_id, separator, frame_number = stem.rpartition("_f")
        if separator == "" or not frame_number.isdigit():
            return stem, None
        else:
            return video_id, int(frame_number)

    def _validate_dataset_path(self) -> None:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")
        if not self.dataset_path.is_dir():
            raise ValueError(f"Dataset path must be a directory: {self.dataset_path}")

    def _domain_paths(self) -> list[Path]:
        return sorted(path for path in self.dataset_path.iterdir() if path.is_dir())

    def _image_paths(self, domain_path: Path) -> list[Path]:
        return sorted(path for path in domain_path.iterdir() if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS)
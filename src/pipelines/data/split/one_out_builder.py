from src.pipelines.data.manifest.manifest_model import ManifestModel
from src.pipelines.data.split.split_model import SplitModel, OneOutSplitModel
from src.loaders.config import SplitConfig, ExperimentConfig
from .builder import SplitBuilder


class OneOutSplitBuilder(SplitBuilder):

    def __init__(self, manifest: ManifestModel, split_config: SplitConfig, experiment_config: ExperimentConfig):
        super().__init__(manifest, split_config)

        self.real_domain = experiment_config.real_domain
        self.fake_domains = experiment_config.fake_domains
        self.held_out_domain = experiment_config.held_out_domain

        self.source_domains = {
            experiment_config.real_domain,
            *[domain for domain in experiment_config.fake_domains if domain != experiment_config.held_out_domain],
        }

        self.target_domains = {
            experiment_config.real_domain,
            experiment_config.held_out_domain,
        }

    def build_split(self) -> OneOutSplitModel:

        self._validate_ratios()
        self._validate_config()

        graph = self._build_graph()
        clusters = self._build_clusters(graph)
        train_ids, validation_ids, test_ids = self._build_splits(clusters)
        return self._build_one_out_split(train_ids, validation_ids, test_ids)

    def _build_one_out_split(self,
        train_ids: set[str],
        validation_ids: set[str],
        test_ids: set[str],
    ) -> OneOutSplitModel:

        source = SplitModel(train=[], validation=[], test=[])
        target = SplitModel(train=[], validation=[], test=[])

        for record in self.manifest.records:
            video_ids = self._extract_id_from_video_id(record.video_id)

            if all(id in train_ids for id in video_ids):
                if record.domain in self.source_domains:
                    source.train.append(record)

            elif all(id in validation_ids for id in video_ids):
                if record.domain in self.source_domains:
                    source.validation.append(record)

            elif all(id in test_ids for id in video_ids):
                if record.domain in self.source_domains:
                    source.test.append(record)
                if record.domain in self.target_domains:
                    target.test.append(record)

        return OneOutSplitModel(source=source, target=target)

    def _validate_config(self) -> None:

        if self.real_domain in self.fake_domains:
            raise ValueError(f"Real domain {self.real_domain} must not be in fake domains {self.fake_domains}")
        if self.held_out_domain not in self.fake_domains:
            raise ValueError(f"Held out domain {self.held_out_domain} is not in fake domains {self.fake_domains}")

        manifest_domains = {record.domain for record in self.manifest.records}
        configured_domains = {self.real_domain, *self.fake_domains}
        missing_domains = configured_domains - manifest_domains
        if missing_domains:
            raise ValueError(f"Configured domains are missing from manifest: {missing_domains}")
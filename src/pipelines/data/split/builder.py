from src.pipelines.data.manifest.manifest_model import ManifestModel
from src.pipelines.data.split.split_model import SplitModel
from src.loaders.config import SplitConfig, ExperimentConfig
from src.pipelines.data.split._helper import (
    build_graph,
    build_clusters,
    build_splits,
    validate_ratios,
    validate_config,
    extract_id_from_video_id,
)


class SplitBuilder:

    def __init__(
        self,
        manifest: ManifestModel,
        split_config: SplitConfig,
        experiment_config: ExperimentConfig,
    ):

        self.manifest = manifest
        self.split_config = split_config
        self.experiment_config = experiment_config

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

    def build_split(self) -> SplitModel:
        
        validate_ratios(self.split_config)
        validate_config(
            self.real_domain,
            self.fake_domains,
            self.held_out_domain,
            self.manifest.records,
        )

        graph = build_graph(self.manifest.records)
        clusters = build_clusters(graph)
        train_ids, validation_ids, test_ids = build_splits(clusters, self.split_config)

        match self.experiment_config.protocol:
            case ExperimentConfig.Protocol.BASELINE | ExperimentConfig.Protocol.GENERALIZATION:
                return self._build_generalization_split(train_ids, validation_ids, test_ids)
            case ExperimentConfig.Protocol.ADAPTATION:
                return self._build_adaptation_split(train_ids, validation_ids, test_ids)
            case _:
                raise ValueError(f"Invalid protocol: {self.experiment_config.protocol}")

    def _build_generalization_split(
        self,
        train_ids: set[str],
        validation_ids: set[str],
        test_ids: set[str],
    ) -> SplitModel:

        split = SplitModel(train=[], validation=[], test=[])

        for record in self.manifest.records:
            video_ids = extract_id_from_video_id(record.video_id)

            if all(id in train_ids for id in video_ids):
                if record.domain in self.source_domains:
                    split.train.append(record)

            elif all(id in validation_ids for id in video_ids):
                if record.domain in self.source_domains:
                    split.validation.append(record)

            elif all(id in test_ids for id in video_ids):
                if record.domain in self.target_domains:
                    split.test.append(record)

        return split

    def _build_adaptation_split(
        self,
        train_ids: set[str],
        validation_ids: set[str],
        test_ids: set[str],
    ) -> SplitModel:

        split = SplitModel(train=[], validation=[], test=[])

        for record in self.manifest.records:
            video_ids = extract_id_from_video_id(record.video_id)

            if all(id in train_ids for id in video_ids):
                if record.domain in self.source_domains:
                    split.train.append(record)
                if record.domain == self.held_out_domain:
                    split.train.append(record)
            elif all(id in validation_ids for id in video_ids):
                if record.domain in self.source_domains:
                    split.validation.append(record)
            elif all(id in test_ids for id in video_ids):
                if record.domain in self.target_domains:
                    split.test.append(record)

        return split
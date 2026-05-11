import random
from collections import defaultdict
from src.pipelines.data.manifest.manifest_model import ManifestModel
from src.pipelines.data.split.split_model import SplitModel
from src.loaders.config import SplitConfig

class SplitBuilder:

    # MARK: - Constants

    _VIDEO_NUM = 1000

    # MARK: - Init

    def __init__(self, manifest: ManifestModel, split_config: SplitConfig):
        self.manifest = manifest
        self.split_config = split_config

    # MARK: - Public Methods

    def build_split(self) -> SplitModel:

        self._validate_ratios()

        graph = self._build_graph()
        clusters = self._build_clusters(graph)
        train_ids, validation_ids, test_ids = self._build_splits(clusters)
        split_model = self._build_split_model(train_ids, validation_ids, test_ids)

        return split_model

    # MARK: - Private Methods

    def _build_graph(self) -> dict[str, set[str]]:

        graph = defaultdict(set)
        for record in self.manifest.records:
            video_ids = self._extract_id_from_video_id(record.video_id)
            match len(video_ids):
                case 2:
                    graph[video_ids[0]].add(video_ids[1])
                    graph[video_ids[1]].add(video_ids[0])
                case 1:
                    if video_ids[0] not in graph:
                        graph[video_ids[0]] = set()
                case _:
                    raise ValueError(f"Invalid video id: {record.video_id}")
        return graph

    def _build_splits(self, clusters: list[list[str]]) -> tuple[list[str], list[str], list[str]]:

        random.seed(self.split_config.seed)
        random.shuffle(clusters)

        train_ids = set()
        validation_ids = set()
        test_ids = set()

        total_cluster_num = len(clusters)
        train_limit = int(total_cluster_num * self.split_config.train_ratio)
        validation_limit = int(total_cluster_num * (self.split_config.train_ratio + self.split_config.validation_ratio))

        for i, cluster in enumerate(clusters):
            if i < train_limit:
                train_ids.update(cluster)
            elif i < validation_limit:
                validation_ids.update(cluster)
            else:
                test_ids.update(cluster)

        return train_ids, validation_ids, test_ids
                    
    def _build_clusters(self, graph: dict[str, set[str]]) -> list[list[str]]:

        visited = set()
        clusters = []
        for node in graph.keys():
            if node not in visited:
                cluster = []
                queue = [node]
                while queue:
                    current = queue.pop(0)
                    if current not in visited:
                        visited.add(current)
                        cluster.append(current)
                        queue.extend(graph[current])
                clusters.append(cluster)
        return clusters

    def _build_split_model(self, train_ids: set[str], validation_ids: set[str], test_ids: set[str]) -> SplitModel:

        split_model = SplitModel(
            train=[],
            validation=[],
            test=[],
        )

        for record in self.manifest.records:
            video_ids = self._extract_id_from_video_id(record.video_id)
            if all(id in train_ids for id in video_ids):
                split_model.train.append(record)
            elif all(id in validation_ids for id in video_ids):
                split_model.validation.append(record)
            elif all(id in test_ids for id in video_ids):
                split_model.test.append(record)
            else:
                continue

        return split_model

    def _validate_ratios(self) -> None:

        values = [
            self.split_config.train_ratio,
            self.split_config.validation_ratio,
            self.split_config.test_ratio,
        ]
        if any(value <= 0 for value in values):
            raise ValueError("Split ratios must be greater than zero.")
        if round(sum(values), 10) != 1:
            raise ValueError("Split ratios must sum to 1.")

    def _extract_id_from_video_id(self, video_id: str) -> list[str]:
        return video_id.split("_")
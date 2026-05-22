import random
from collections import defaultdict
from src.pipelines.data.image_record_model import ImageRecordModel
from src.loaders.config import SplitConfig

def build_graph(records: list[ImageRecordModel]) -> dict[str, set[str]]:

    graph = defaultdict(set)
    for record in records:
        video_ids = extract_id_from_video_id(record.video_id)
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

def build_splits(clusters: list[list[str]], split_config: SplitConfig) -> tuple[list[str], list[str], list[str]]:

    random.seed(split_config.seed)
    random.shuffle(clusters)

    train_ids = set()
    validation_ids = set()
    test_ids = set()

    total_cluster_num = len(clusters)
    train_limit = int(total_cluster_num * split_config.train_ratio)
    validation_limit = int(total_cluster_num * (split_config.train_ratio + split_config.validation_ratio))

    for i, cluster in enumerate(clusters):
        if i < train_limit:
            train_ids.update(cluster)
        elif i < validation_limit:
            validation_ids.update(cluster)
        else:
            test_ids.update(cluster)

    return train_ids, validation_ids, test_ids

def build_clusters(graph: dict[str, set[str]]) -> list[list[str]]:

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

def validate_ratios(split_config: SplitConfig) -> None:

    values = [
        split_config.train_ratio,
        split_config.validation_ratio,
        split_config.test_ratio,
    ]
    if any(value <= 0 for value in values):
        raise ValueError("Split ratios must be greater than zero.")
    if round(sum(values), 10) != 1:
        raise ValueError("Split ratios must sum to 1.")

def validate_config(
    real_domain: str,
    fake_domains: list[str],
    held_out_domain: str,
    records: list[ImageRecordModel],
) -> None:

    if real_domain in fake_domains:
        raise ValueError(f"Real domain {real_domain} must not be in fake domains {fake_domains}")
    if held_out_domain not in fake_domains:
        raise ValueError(f"Held out domain {held_out_domain} is not in fake domains {fake_domains}")

    manifest_domains = {record.domain for record in records}
    configured_domains = {real_domain, *fake_domains}
    missing_domains = configured_domains - manifest_domains
    if missing_domains:
        raise ValueError(f"Configured domains are missing from manifest: {missing_domains}")

def extract_id_from_video_id(video_id: str) -> list[str]:
    return video_id.split("_")
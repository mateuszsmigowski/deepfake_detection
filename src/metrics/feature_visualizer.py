import torch
import numpy as np
from enum import StrEnum
import os
import tempfile
import umap
from sklearn.manifold import TSNE

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

class FeatureVisualizer:

    class ReductionMethod(StrEnum):
        UMAP = "umap"
        TSNE = "tsne"

    def __init__(self, output_dir: Path, visualization_interval: int):

        self._output_dir = output_dir / "feature_visualization"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._features: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []
        self._domain_names: list[str] = []
        self._method = FeatureVisualizer.ReductionMethod.UMAP
        self._visualization_interval = visualization_interval

    def add_features(
        self,
        epoch: int,
        features: torch.Tensor,
        labels: torch.Tensor,
        domain_names: list[str],
    ):

        if not self._should_visualize(epoch):
            return

        if len(domain_names) != features.size(0):
            raise ValueError("Number of domain names must match the number of features.")

        self._features.append(features.detach().cpu())
        self._labels.append(labels.detach().cpu())
        self._domain_names.extend(domain_names)

    def flush(self, epoch: int, split: str):

        if not self._should_visualize(epoch):
            return

        if not self._features:
            return

        features = torch.cat(self._features, dim=0).numpy()
        labels = torch.cat(self._labels, dim=0).numpy()
        domain_names = np.array(self._domain_names)

        embeddings = self._reduce(features)
        self._plot(embeddings, labels, domain_names, epoch, split)
        self.reset()

    def reset(self):
        self._features.clear()
        self._labels.clear()
        self._domain_names.clear()

    def _should_visualize(self, epoch: int) -> bool:
        return epoch == 1 or epoch % self._visualization_interval == 0

    def _reduce(self, features: np.ndarray) -> np.ndarray:
        if features.shape[0] < 2:
            raise ValueError("Feature visualization requires at least two samples.")

        match self._method:
            case FeatureVisualizer.ReductionMethod.UMAP:
                return umap.UMAP(
                    n_components=2,
                    n_neighbors=min(15, max(2, features.shape[0] - 1)),
                    min_dist=0.1,
                    metric="cosine",
                    random_state=100,
                    n_jobs=1,
                ).fit_transform(features)
            case FeatureVisualizer.ReductionMethod.TSNE:
                return TSNE(
                    n_components=2,
                    perplexity=min(30, features.shape[0] - 1),
                    learning_rate="auto",
                    init="pca",
                    random_state=100,
                ).fit_transform(features)

    def _plot(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        domain_names: np.ndarray,
        epoch: int,
        split: str,
    ):
        label_names = {0.0: "Real", 1.0: "Fake"}
        label_colors = {0.0: "blue", 1.0: "red"}

        unique_domains = sorted(set(domain_names))
        cmap = plt.get_cmap("tab10")
        domain_colors = {
            domain: cmap(i) for i, domain in enumerate(unique_domains)}

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for label_value in np.unique(labels):
            mask = labels == label_value
            axes[0].scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                s=12,
                alpha=0.7,
                c=label_colors.get(float(label_value), "black"),
                label=label_names.get(float(label_value), str(label_value)),
            )
        
        axes[0].set_title("Color == class label")
        axes[0].legend()
        axes[0].set_xlabel("dim 1")
        axes[0].set_ylabel("dim 2")

        for domain_name in unique_domains:
            mask = domain_names == domain_name
            axes[1].scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                s=12,
                alpha=0.7,
                c=[domain_colors[domain_name]],
                label=domain_name,
            )
        
        axes[1].set_title("Color = domain")
        axes[1].legend()
        axes[1].set_xlabel("dim 1")
        axes[1].set_ylabel("dim 2")

        fig.suptitle(f"{self._method.value.upper()} Epoch {epoch}, Split {split}")
        fig.tight_layout()

        output_path = self._output_dir / f"epoch_{epoch}_{split}.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

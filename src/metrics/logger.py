import json
import torch
from datetime import datetime
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
from src.loaders.config import ConfigModel

class MetricsLogger:

    def __init__(self, config: ConfigModel):

        self.config = config
        self.run_dir = (
            config.outputs.runs_path / 
            config.experiment.name /
            f"seed-{config.runtime.seed}" /
            f"_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.best_metric: float | None = None

        self.run_dir.mkdir(parents=True, exist_ok=True)
        if config.outputs.save_checkpoints:
            self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self.save_config()

    def save_config(self):

        config_path = self.run_dir / "config.json"
        with config_path.open("w", encoding="utf-8") as file:
            json.dump(self._jsonable(asdict(self.config)), file, indent=4)

    def log_metrics(self, split: str, metrics: dict[str, float], epoch: int | None = None):

        self._print_metrics(split, metrics, epoch)

        if not self.config.outputs.save_metrics:
            return

        row = {
            "experiment": self.config.experiment.name,
            "mode": self.config.experiment.mode,
            "split": split,
            "epoch": epoch,
            "seed": self.config.runtime.seed,
            "held_out_domain": self.config.experiment.held_out_domain,
            **{f"{split}_{key}": value for key, value in metrics.items()}
        }

        with self.metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(self._jsonable(row), ensure_ascii=False) + "\n")

    def save_best_checkpoint(
        self,
        model: torch.nn.Module,
        metrics: dict[str, float],
        split: str,
        epoch: int,
    ):
        
        if not self.config.outputs.save_checkpoints:
            return

        metric_name = self.config.outputs.checkpoint_metric
        metric_value = self._resolve_metric(metric_name, split, metrics)

        if metric_value is None:
            raise ValueError(f"Checkpoint metric '{metric_name}' not found in {split} metrics.")

        if self.best_metric is None or self._is_better(metric_name, metric_value):
            self.best_metric = metric_value
            torch.save(
                {
                    "epoch": epoch,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "model_state_dict": model.state_dict(),
                    "config": self._jsonable(asdict(self.config)),
                },
                self.checkpoints_dir / "best.pt"
            )

    def load_best_checkpoint(
        self,
        model: torch.nn.Module,
        device: torch.device,
    ) -> dict[str, Any] | None:

        if not self.config.outputs.save_checkpoints:
            return None

        checkpoint_path = self.checkpoints_dir / "best.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Best checkpoint not found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint

    def _resolve_metric(
        self,
        metric_name: str,
        split: str,
        metrics: dict[str, float],
    ) -> float | None:

        prefixed = f"{split}_{metric_name}"

        if prefixed in metrics:
            return metrics[prefixed]
        if metric_name.startswith(f"{split}_"):
            return metrics.get(metric_name.removeprefix(f"{split}_"))
        return metrics.get(metric_name)

    def _is_better(
        self,
        metric_name: str,
        value: float,
    ) -> bool:

        if self.best_metric is None:
            return True
        if "loss" in metric_name:
            return value < self.best_metric
        return value > self.best_metric

    def _print_metrics(
        self,
        split: str,
        metrics: dict[str, float],
        epoch: int | None = None,
    ):
        prefix = f"{split}"
        if epoch is not None:
            prefix += f" (Epoch {epoch})"
        print(prefix, end=" ")
        for key, value in metrics.items():
            print(f"-- {key}: {value:.4f}", end=" ")
        print()

    @classmethod
    def _jsonable(cls, value: Any) -> Any:

        if isinstance(value, Path):
            return str(value)
        if is_dataclass(value):
            return cls._jsonable(asdict(value))
        if isinstance(value, dict):
            return {key: cls._jsonable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._jsonable(item) for item in value]
        return value
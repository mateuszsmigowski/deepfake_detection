from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from types import SimpleNamespace
from contextlib import redirect_stdout
import io
import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", gettempdir())

import torch
from torch.utils.data import DataLoader

from src.loaders.config import (
    ConfigModel,
    DataConfig,
    DomainAdaptationConfig,
    ExperimentConfig,
    ManifestConfig,
    ModelConfig,
    OutputsConfig,
    RuntimeConfig,
    SplitConfig,
    TrainingConfig,
)
from src.training.dann.adaptation.adaptation_trainer import DANNAdaptationTrainer


class DummyDANN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, images, grl_lambda=None, domain_mask=None, skip_domain=False):
        batch_size = images.size(0)
        label_logits = self.weight.expand(batch_size)
        features = self.weight.expand(batch_size, 2)

        if skip_domain:
            domain_logits = self.weight.new_empty((0, 2))
        else:
            domain_batch_size = int(domain_mask.sum().item()) if domain_mask is not None else batch_size
            domain_logits = torch.stack(
                [
                    self.weight.expand(domain_batch_size),
                    (-self.weight).expand(domain_batch_size),
                ],
                dim=1,
            )

        return SimpleNamespace(
            label_logits=label_logits,
            domain_logits=domain_logits,
            features=features,
        )


class RecordingCriterion:
    def __init__(self):
        self.labels = []

    def __call__(self, logits, labels):
        self.labels.append(labels.detach().cpu().clone())
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)


class RecordingDomainCriterion:
    def __init__(self):
        self.domains = []

    def __call__(self, logits, domains):
        self.domains.append(domains.detach().cpu().clone())
        return torch.nn.functional.cross_entropy(logits, domains)


class NoOpFeatureVisualizer:
    def reset(self):
        pass

    def add_features(self, *args, **kwargs):
        pass

    def flush(self, *args, **kwargs):
        pass


def _config(tmp_path):
    return ConfigModel(
        experiment=ExperimentConfig(
            name="test",
            mode=ExperimentConfig.Mode.DANN,
            protocol=ExperimentConfig.Protocol.ADAPTATION,
            real_domain="Original",
            fake_domains=["Deepfakes", "Face2Face"],
            held_out_domain="Face2Face",
        ),
        data=DataConfig(
            dataset_path=tmp_path,
            train_source_real=1,
            train_source_fake=1,
            val_source_real=1,
            val_source_fake=1,
            train_target_fake=1,
        ),
        runtime=RuntimeConfig(
            seed=1,
            device=RuntimeConfig.Device.CPU,
            deterministic=True,
            num_workers=0,
        ),
        model=ModelConfig(
            architecture=ModelConfig.Architecture.RESNET18,
            pretrained=False,
            freeze_backbone=False,
        ),
        manifest=ManifestConfig(path=tmp_path / "manifest.csv"),
        training=TrainingConfig(
            epochs=1,
            batch_size=2,
            learning_rate=0.0,
            weight_decay=0.0,
            validation_interval=1,
            early_stopping_patience=None,
            pos_weight=False,
        ),
        split=SplitConfig(
            train_ratio=0.7,
            validation_ratio=0.15,
            test_ratio=0.15,
            seed=1,
            path=tmp_path / "splits",
        ),
        domain_adaptation=DomainAdaptationConfig(
            gradient_reversal_lambda=1.0,
            domain_loss_weight=0.1,
        ),
        outputs=OutputsConfig(
            runs_path=tmp_path / "runs",
            save_checkpoints=False,
            save_metrics=False,
            checkpoint_metric="validation_balanced_accuracy",
        ),
    )


def _batch(label, domain, domain_name):
    return {
        "image": torch.ones(1, 2, 2),
        "label": torch.tensor(label, dtype=torch.float32),
        "domain": torch.tensor(domain, dtype=torch.long),
        "metadata": {"domain": domain_name},
    }


def _loader(samples, batch_size=2):
    return DataLoader(samples, batch_size=batch_size, shuffle=False)


class DANNAdaptationTrainerTests(unittest.TestCase):
    def test_adaptation_label_loss_uses_only_source_labels(self):
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            source_loader = _loader(
                [
                    _batch(0, -1, "Original"),
                    _batch(1, 0, "Deepfakes"),
                ],
            )
            target_loader = _loader([_batch(1, 1, "Face2Face")], batch_size=1)
            evaluation_loader = _loader(
                [
                    _batch(0, -1, "Original"),
                    _batch(1, 1, "Face2Face"),
                ],
            )

            trainer = DANNAdaptationTrainer(
                _config(tmp_path),
                DummyDANN(),
                source_loader,
                target_loader,
                evaluation_loader,
                evaluation_loader,
            )
            criterion = RecordingCriterion()
            domain_criterion = RecordingDomainCriterion()
            trainer.label_criterion = criterion
            trainer.domain_criterion = domain_criterion
            trainer.feature_visualizer = NoOpFeatureVisualizer()

            with redirect_stdout(io.StringIO()):
                trainer.run()

            self.assertEqual(criterion.labels[0].tolist(), [0.0, 1.0])
            self.assertEqual(domain_criterion.domains[0].tolist(), [0, 1])

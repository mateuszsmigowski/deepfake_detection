from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.loaders.config import ExperimentConfig, SplitConfig
from src.pipelines.data.manifest.manifest_model import ManifestModel
from src.pipelines.data.split.manager import SplitManager


def _experiment_config():
    return ExperimentConfig(
        name="test",
        mode=ExperimentConfig.Mode.BASELINE,
        protocol=ExperimentConfig.Protocol.BASELINE,
        real_domain="Original",
        fake_domains=["Deepfakes", "Face2Face"],
        held_out_domain="Face2Face",
    )


class SplitManagerTests(unittest.TestCase):
    def test_split_cache_path_includes_seed_and_ratios(self):
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            manifest = ManifestModel(records=[])
            experiment_config = _experiment_config()

            first = SplitManager(
                manifest,
                SplitConfig(
                    train_ratio=0.7,
                    validation_ratio=0.15,
                    test_ratio=0.15,
                    seed=1,
                    path=tmp_path,
                ),
                experiment_config,
            )
            second = SplitManager(
                manifest,
                SplitConfig(
                    train_ratio=0.6,
                    validation_ratio=0.2,
                    test_ratio=0.2,
                    seed=2,
                    path=tmp_path,
                ),
                experiment_config,
            )

            self.assertNotEqual(first._split_dir, second._split_dir)
            self.assertIn(
                "seed-1_train-0.7_val-0.15_test-0.15",
                first._split_dir.as_posix(),
            )
            self.assertIn(
                "seed-2_train-0.6_val-0.2_test-0.2",
                second._split_dir.as_posix(),
            )

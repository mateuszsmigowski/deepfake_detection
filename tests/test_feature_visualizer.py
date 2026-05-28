from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", gettempdir())

import numpy as np
import torch

from src.metrics.feature_visualizer import FeatureVisualizer


class FeatureVisualizerTests(unittest.TestCase):
    def test_visualizer_collects_and_flushes_only_on_interval(self):
        with TemporaryDirectory() as directory:
            visualizer = FeatureVisualizer(Path(directory), visualization_interval=5)
            plotted = []

            visualizer._reduce = lambda features: np.zeros((features.shape[0], 2))
            visualizer._plot = lambda *args: plotted.append(args)

            features = torch.ones(2, 4)
            labels = torch.tensor([0.0, 1.0])
            domains = ["Original", "Deepfakes"]

            visualizer.add_features(1, features, labels, domains)
            visualizer.flush(1, "train")
            self.assertEqual(len(plotted), 1)
            self.assertEqual(plotted[0][3], 1)
            self.assertEqual(plotted[0][4], "train")

            visualizer.add_features(4, features, labels, domains)
            visualizer.flush(4, "train")
            self.assertEqual(len(plotted), 1)

            visualizer.add_features(5, features, labels, domains)
            visualizer.flush(5, "train")

            self.assertEqual(len(plotted), 2)
            self.assertEqual(plotted[1][3], 5)
            self.assertEqual(plotted[1][4], "train")

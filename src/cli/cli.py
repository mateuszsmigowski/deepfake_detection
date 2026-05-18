import argparse
from pathlib import Path
from dataclasses import dataclass

from src.experiments import run_experiment
from src.orchestrator import orchestrate
from src.loaders.config import load_config, ConfigModel

@dataclass(frozen=True)
class Arguments:
    config: Path
    experiment: bool

# MARK: - Cli

class Cli:

    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def run(self):

        arguments: Arguments = self._parse_arguments()
        config: ConfigModel = load_config(arguments.config)

        if arguments.experiment:
            run_experiment(config)
        else:
            orchestrate(config)

    def _parse_arguments(self) -> Arguments:

        self.parser.add_argument(
            "config",
            type=Path,
            help="Path to YAML config file.",
        )
        self.parser.add_argument(
            "--experiment",
            action="store_true",
            help="Run experiment.",
        )
        return self.parser.parse_args()
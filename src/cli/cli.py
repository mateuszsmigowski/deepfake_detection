import argparse
from pathlib import Path
from dataclasses import dataclass
from src.orchestrator import orchestrate
from src.loaders.config import load_config, ConfigModel

@dataclass(frozen=True)
class Arguments:
    config: Path

# MARK: - Cli

class Cli:

    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def run(self):

        arguments: Arguments = self._parse_arguments()
        config: ConfigModel = load_config(arguments.config)
        orchestrate(config)

    def _parse_arguments(self) -> Arguments:

        self.parser.add_argument("config", type=Path, help="Path to YAML config file.")
        return self.parser.parse_args()
from pathlib import Path
from typing import NamedTuple

import tomli

PARENT_DIR = Path(__file__).parent
PROJECT_DIR = PARENT_DIR.parent.parent

with open(PARENT_DIR / "config.toml", "rb") as fb:
    config = tomli.load(fb)

with open(PARENT_DIR / "instruction_template.txt") as f:
    instruction_template = f.read()


class Paths(NamedTuple):
    chunks: Path
    attempts: Path
    plots: Path
    results: Path


class Config:
    data_path = PROJECT_DIR / "data"
    exclude_files_above = config["exclude_files_above"]
    exclude_extensions = config["exclude_extensions"]
    instruction_template = instruction_template

    paths = Paths(
        chunks=data_path / "chunks",
        attempts=data_path / "attempts",
        plots=data_path / "plots",
        results=data_path / "results",
    )


# Create all directories in the config if they don't exist
for path in Config.paths:
    path.mkdir(parents=True, exist_ok=True)

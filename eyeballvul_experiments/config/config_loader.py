from pathlib import Path
from typing import NamedTuple

import tomli

PARENT_DIR = Path(__file__).parent
PROJECT_DIR = PARENT_DIR.parent.parent

with open(PARENT_DIR / "config.toml", "rb") as f:
    config = tomli.load(f)


class Paths(NamedTuple):
    chunks: Path
    plots: Path
    results: Path


class Config:
    data_path = PROJECT_DIR / "data"

    paths = Paths(
        chunks=data_path / "chunks",
        plots=data_path / "plots",
        results=data_path / "results",
    )


# Create all directories in the config if they don't exist
for path in Config.paths:
    path.mkdir(parents=True, exist_ok=True)

from pathlib import Path
from typing import NamedTuple

import tomli

from eyeballvul_experiments.llm_gateway.gateway_interface import GatewayInterface
from eyeballvul_experiments.llm_gateway.litellm_gateway import LiteLLMGateway

PARENT_DIR = Path(__file__).parent
PROJECT_DIR = PARENT_DIR.parent.parent

with open(PARENT_DIR / "config.toml", "rb") as fb:
    config = tomli.load(fb)

with open(PARENT_DIR / "instruction_template.txt") as f:
    instruction_template = f.read()

with open(PARENT_DIR / "cwe_list.txt") as f:
    cwe_list = f.read()


class Paths(NamedTuple):
    chunks: Path
    attempts: Path
    plots: Path
    results: Path
    cache: Path


class Config:
    data_path = PROJECT_DIR / "data"
    exclude_files_above = config["exclude_files_above"]
    exclude_extensions = config["exclude_extensions"]
    instruction_template = instruction_template
    cwe_list = cwe_list
    gateway: GatewayInterface = LiteLLMGateway()

    paths = Paths(
        chunks=data_path / "chunks",
        attempts=data_path / "attempts",
        plots=data_path / "plots",
        results=data_path / "results",
        cache=data_path / "cache",
    )


# Create all directories in the config if they don't exist
for path in Config.paths:
    path.mkdir(parents=True, exist_ok=True)

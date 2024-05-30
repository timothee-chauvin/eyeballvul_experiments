import json
import logging
from dataclasses import dataclass

from typeguard import typechecked

from eyeballvul_experiments.config.config_loader import Config
from eyeballvul_experiments.util import get_str_weak_hash


@dataclass
class File:
    # Filename relative to the repository root.
    filename: str
    contents: str

    def format(self) -> str:
        """Format for the model, for inclusion in the formatted chunk."""
        return f"<><><>CONTENTS OF {self.filename}:<><><>\n{self.contents}\n"


@typechecked
class Chunk:
    """Chunk of a repository."""

    def __init__(self, repo_url: str, commit: str, repo_size: int):
        self.files: list[File] = []
        self.repo_url: str = repo_url
        self.commit: str = commit
        self.repo_size: int = repo_size

    def __len__(self) -> int:
        """Length of the chunk as seen by the model."""
        return len(self.format_prefix()) + sum(len(file.format()) for file in self.files)

    def shrink(self) -> list[File]:
        """
        Shrink the chunk by at least one file (starting from the end), and at most 5% of its size.

        Return the list of files removed (kept in order).
        """
        initial_length = len(self)
        removed_files: list[File] = []
        while len(self) > 0.95 * initial_length:
            removed_files.insert(0, self.files.pop())
        logging.info(f"Shrunk chunk {self.get_hash()} from {initial_length} to {len(self)}.")
        return removed_files

    def format_prefix(self) -> str:
        """
        Prefix for the chunk, as seen by the model.

        The percentage always has the same width (6 characters).
        """
        return f"<><><>Fraction of the full repository included below: {self.fraction_included():6.1%}<><><>\n"

    def full_str(self) -> str:
        """Full string representation of the chunk, as seen by the model."""
        return self.format_prefix() + "".join(file.format() for file in self.files)

    def get_hash(self) -> str:
        return get_str_weak_hash(self.full_str())

    def fraction_included(self) -> float:
        return sum(len(file.contents) for file in self.files) / self.repo_size

    def log(self) -> None:
        directory = Config.paths.chunks
        json_dict = {
            "version": "0.1.0",
            "repo_url": self.repo_url,
            "commit": self.commit,
            "length": len(self),
            "fraction_included": self.fraction_included(),
            "filenames": sorted(file.filename for file in self.files),
        }
        with open(directory / f"{self.get_hash()}.json", "w") as f:
            json.dump(json_dict, f, indent=2)
            f.write("\n")

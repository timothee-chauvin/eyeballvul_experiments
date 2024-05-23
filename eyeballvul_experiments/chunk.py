import json
from dataclasses import dataclass

from typeguard import typechecked

from eyeballvul_experiments.config.config_loader import Config
from eyeballvul_experiments.util import get_str_weak_hash


@dataclass
class File:
    filename: str
    contents: str

    def format(self) -> str:
        """Format for the model, for inclusion in the formatted chunk."""
        return f"<><><>CONTENTS OF {self.filename}:<><><>\n{self.contents}\n"


@dataclass
@typechecked
class Chunk:
    """Chunk of a repository."""

    files: list[File]
    repo_url: str
    commit: str
    # What fraction of the full repository is included in the chunk, in terms of file contents.
    fraction_included: float

    def __len__(self) -> int:
        """Length of the chunk as seen by the model."""
        return len(self.format_prefix()) + sum(len(file.format()) for file in self.files)

    def would_overflow(self, new_file: File, max_size: int) -> int:
        """Would adding `new_file` to the chunk exceed the maximum size?"""
        return len(self) + len(new_file.format()) > max_size

    def format_prefix(self) -> str:
        return f"<><><>Fraction of the full repository included below: {self.fraction_included:.1%}<><><>\n"

    def full_str(self) -> str:
        """Full string representation of the chunk, as seen by the model."""
        return self.format_prefix() + "".join(file.format() for file in self.files)

    def get_hash(self) -> str:
        return get_str_weak_hash(self.full_str())

    def log(self) -> None:
        directory = Config.paths.chunks
        json_dict = {
            "version": "0.1.0",
            "repo_url": self.repo_url,
            "commit": self.commit,
            "length": len(self),
            "fraction_included": self.fraction_included,
            "filenames": sorted(file.filename for file in self.files),
        }
        with open(directory / f"{self.get_hash()}.json", "w") as f:
            json.dump(json_dict, f, indent=2)
            f.write("\n")

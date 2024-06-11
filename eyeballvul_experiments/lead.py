from typing import Literal

from pydantic import BaseModel, model_validator


class Lead(BaseModel):
    headline: str
    analysis: str
    classification: Literal["very promising", "slightly promising", "not promising"]
    cwe: str | None = None
    function_names: list[str] = []
    filenames: list[str] = []

    @model_validator(mode="after")
    def trim_lists(self) -> "Lead":
        if len(self.function_names) > 3:
            self.function_names = self.function_names[:3]
        if len(self.filenames) > 3:
            self.filenames = self.filenames[:3]
        return self

    def format(self) -> str:
        """Format the lead for the scoring model."""
        return (
            f"headline: {self.headline}\n"
            f"analysis: {self.analysis}\n"
            f"cwe: {self.cwe}\n"
            f"function_names: {self.function_names}\n"
            f"filenames: {self.filenames}"
        )

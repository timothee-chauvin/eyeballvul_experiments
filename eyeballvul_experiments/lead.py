from typing import Literal

from pydantic import BaseModel


class Lead(BaseModel):
    headline: str
    analysis: str
    classification: Literal["very promising", "slightly promising", "not promising"]

    def format(self) -> str:
        """Format the lead for the scoring model."""
        return f"- headline: {self.headline}\nanalysis: |\n{self.analysis}\n"

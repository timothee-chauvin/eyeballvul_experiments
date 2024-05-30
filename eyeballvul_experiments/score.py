from datetime import datetime

from pydantic import BaseModel


class Score(BaseModel):
    stats: dict[str, int] | dict[str, int | dict[str, int]]
    mapping: dict[int, str]
    date: datetime

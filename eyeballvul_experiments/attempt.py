import json
import logging
import re
from datetime import datetime
from typing import NamedTuple

import yaml
from eyeballvul import score
from litellm import model_cost
from pydantic import BaseModel
from typeguard import typechecked

from eyeballvul_experiments.config.config_loader import Config
from eyeballvul_experiments.lead import Lead
from eyeballvul_experiments.score import Score
from eyeballvul_experiments.util import get_str_weak_hash


class Response(BaseModel):
    content: str
    date: datetime


class Usage(NamedTuple):
    input_tokens: int
    output_tokens: int

    # Not overwriting __add__ because it would have a different signature than the usual one for tuples, so would be confusing.
    def plus(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class Attempt(BaseModel):
    """
    Attempt of a model on a given commit.

    Contains a list of chunks and associated responses, as well as the leads parsed from the
    responses and a list of scores (a list, because the same attempt may be scored multiple times).
    """

    commit: str
    repo_url: str
    model: str
    chunk_hashes: list[str] = []
    responses: list[Response] = []
    leads: list[Lead] = []
    scores: list[Score] = []
    # Note: usage and score only apply to the initial queries, not the scoring queries.
    usage: Usage = Usage(0, 0)
    # Cost in USD as of running, based on usage.
    cost: float = 0.0
    version: str = "0.1.0"

    def parse(self):
        """Parse `self.responses` to populate `self.leads`, overwriting it if it already exists."""
        self.leads = []
        for response in self.responses:
            self.leads += self._parse_response(response)

    @staticmethod
    def _parse_response(response: Response) -> list[Lead]:
        try:
            yaml_content = extract_yaml_from_str(response.content)
        except ValueError:
            logging.warning("Error extracting YAML from model response")
            return []

        leads_yaml = yaml_content["leads"]
        leads: list[Lead] = []
        for i, lead_yaml in enumerate(leads_yaml):
            try:
                lead = Lead(**lead_yaml)
                leads.append(lead)
            except ValueError as e:
                logging.warning(f"Error parsing lead #{i}: {lead_yaml}")
                logging.warning(e)
        return leads

    def add_score(self):
        """
        Compute and add a score to `self.scores`.

        Only leads marked as `very promising` are considered.
        """
        kept_leads = [lead for lead in self.leads if lead.classification == "very promising"]
        stats, mapping = score(self.commit, [lead.format() for lead in kept_leads])
        self.scores.append(Score(stats=stats, mapping=mapping, date=datetime.now()))

    def update_usage_and_cost(self, usage: Usage):
        self.usage = self.usage.plus(usage)
        self.cost = (
            model_cost[self.model]["input_cost_per_token"] * self.usage.input_tokens
            + model_cost[self.model]["output_cost_per_token"] * self.usage.output_tokens
        )

    def get_hash(self):
        """Hash based only on the responses, so it doesn't change when e.g. new scores are added."""
        return get_str_weak_hash(
            json.dumps([response.model_dump(mode="json") for response in self.responses])
        )

    def log(self):
        with open(Config.paths.attempts / f"{self.get_hash()}.json", "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2)
            f.write("\n")


@typechecked
def extract_yaml_from_str(s: str) -> dict:
    """
    Extract a yaml dictionary from a string.

    If there's a ```yaml ...``` code block, its contents are parsed. If there isn't, we try to parse
    the whole string as yaml.

    :params:
    :param s: The string to extract the yaml dictionary from.
    :returns: The extracted yaml dictionary.
    :raises: ValueError if the string can't be parsed as yaml.
    """

    try:
        yaml_part = re.search(r"```yaml\s*(.*?)\s*```", s, re.DOTALL)
        match yaml_part:
            case None:
                s = s.replace("`", "")
                yaml_content = yaml.safe_load(s)
            case _:
                content_fixed = yaml_part.group(1).replace("`", "")
                yaml_content = yaml.safe_load(content_fixed)
        if not isinstance(yaml_content, dict):
            raise ValueError("Yaml content is not a dictionary.")

        return yaml_content
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing yaml from string: {e}")

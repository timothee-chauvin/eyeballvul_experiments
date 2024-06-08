"""Interface to an LLM gateway."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


class ContextWindowExceededError(Exception):
    pass


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    cost: float


@dataclass
class Message:
    content: str
    role: str


@dataclass
class Choice:
    message: Message


@dataclass
class Response:
    choices: list[Choice]
    usage: Usage


class GatewayInterface(ABC):
    @abstractmethod
    async def acompletion(self, model: str, messages: list[dict], num_retries: int = 0) -> Response:
        """
        Generate completions using the gateway.

        Arguments:
        - settings: the settings to use for generation

        Returns:
        - the response from the gateway
        """
        raise NotImplementedError()

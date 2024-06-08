from litellm import ContextWindowExceededError as LiteLLMContextWindowExceededError
from litellm import acompletion, cost_per_token

from eyeballvul_experiments.llm_gateway.gateway_interface import (
    Choice,
    ContextWindowExceededError,
    GatewayInterface,
    Message,
    Response,
    Usage,
)


class LiteLLMGateway(GatewayInterface):
    @staticmethod
    async def acompletion(model: str, messages: list[dict], num_retries: int = 0) -> Response:
        try:
            response = await acompletion(model=model, messages=messages, num_retries=num_retries)
        except LiteLLMContextWindowExceededError as e:
            raise ContextWindowExceededError(e)
        prompt_cost, completion_cost = cost_per_token(
            model=model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )
        return Response(
            choices=[
                Choice(message=Message(content=choice.message.content, role=choice.message.role))
                for choice in response.choices
            ],
            usage=Usage(
                completion_tokens=response.usage.completion_tokens,
                prompt_tokens=response.usage.prompt_tokens,
                cost=prompt_cost + completion_cost,
            ),
        )

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
    async def acompletion(
        model: str, messages: list[dict], num_retries: int = 0, safety_settings: list | None = None
    ) -> Response:
        kwargs = {
            "model": model,
            "messages": messages,
            "num_retries": num_retries,
        }
        if safety_settings is not None:
            kwargs["safety_settings"] = safety_settings
        try:
            response = await acompletion(**kwargs)
        except LiteLLMContextWindowExceededError as e:
            raise ContextWindowExceededError(e)
        model_for_cost = model.removeprefix("gemini/")
        prompt_cost, completion_cost = cost_per_token(
            model=model_for_cost,
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

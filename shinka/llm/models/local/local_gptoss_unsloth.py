import backoff
import openai
import logging
from ..result import QueryResult

logger = logging.getLogger(__name__)


def backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.info(
            f"Local model - Retry {details['tries']} due to error: {exc}. Waiting {details['wait']:0.1f}s..."
        )


@backoff.on_exception(
    backoff.expo,
    (
        openai.APIConnectionError,
        openai.APIStatusError,
        openai.RateLimitError,
        openai.APITimeoutError,
    ),
    max_tries=10,
    max_value=20,
    on_backoff=backoff_handler,
)

def query_local_gptoss_unsloth(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model=None,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    """Query a local OpenAI-compatible model with reasoning extraction."""
    new_msg_history = msg_history + [{"role": "user", "content": msg}]
    
    if output_model is not None:        raise ValueError("Local model does not support structured output.")

    #print(kwargs)
    
    # Replace max tokens by tokens
    if "max_output_tokens" in kwargs:
        kwargs["max_tokens"] = kwargs.pop("max_output_tokens")

    # Send request to local OpenAI-compatible API
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            *new_msg_history,
        ],
        **kwargs,
    )

    choice = response.choices[0].message
    content = choice.content
    thought = getattr(choice, "reasoning_content", "")

    # Update message history
    new_msg_history.append({"role": "assistant", "content": content})

    # Extract token usage if available
    input_tokens = getattr(response.usage, "prompt_tokens", 0)
    total_tokens = getattr(response.usage, "total_tokens", 0)
    output_tokens = getattr(response.usage, "completion_tokens", total_tokens - input_tokens)

    # Local models typically have no cost
    input_cost = 0.0
    output_cost = 0.0

    # Construct result
    result = QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        model_name=model,
        kwargs=kwargs,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=0.0,
        input_cost=input_cost,
        output_cost=output_cost,
        thought=thought,
        model_posteriors=model_posteriors,
    )

    return result

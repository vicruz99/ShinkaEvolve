from typing import Any, Tuple
import os
import anthropic
import openai
import instructor
from pathlib import Path
import re                                                                                                   # ADDED THIS LINE       
from dotenv import load_dotenv
from google import genai
from .providers.pricing import get_provider


env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

TIMEOUT = 600


def get_client_llm(model_name: str, structured_output: bool = False) -> Tuple[Any, str]:
    """Get the client and model for the given model name.

    Args:
        model_name (str): The name of the model to get the client.

    Raises:
        ValueError: If the model is not supported.

    Returns:
        The client and model for the given model name.
    """
    # print(f"Getting client for model {model_name}")
    provider = get_provider(model_name)

    if provider == "anthropic":
        client = anthropic.Anthropic(timeout=TIMEOUT)  # 10 minutes
        if structured_output:
            client = instructor.from_anthropic(
                client, mode=instructor.mode.Mode.ANTHROPIC_JSON
            )
    elif provider == "bedrock":
        client = anthropic.AnthropicBedrock(
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_region=os.getenv("AWS_REGION_NAME"),
            timeout=TIMEOUT,  # 10 minutes
        )
        if structured_output:
            client = instructor.from_anthropic(
                client, mode=instructor.mode.Mode.ANTHROPIC_JSON
            )
    elif provider == "openai":
        client = openai.OpenAI(timeout=TIMEOUT)  # 10 minutes
        if structured_output:
            client = instructor.from_openai(client, mode=instructor.Mode.TOOLS_STRICT)

    #### AZURE OPENAI ####
    elif model_name.startswith("azure-"):
        # get rid of the azure- prefix
        model_name = model_name.split("azure-")[-1]
        # https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle?view=foundry-classic&tabs=python#api-evolution
        client = openai.AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_API_ENDPOINT") + "openai/v1/",
            timeout=TIMEOUT,  # 10 minutes
        )
        if structured_output:
            client = instructor.from_openai(client, mode=instructor.Mode.TOOLS_STRICT)
    elif provider == "deepseek":
        client = openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
            timeout=TIMEOUT,  # 10 minutes
        )
        if structured_output:
            client = instructor.from_openai(client, mode=instructor.Mode.MD_JSON)
    elif provider == "google":
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        if structured_output:
            client = instructor.from_openai(
                client,
                mode=instructor.Mode.GEMINI_JSON,
            )
    
    #### LOCAL GPTOSS UNSLOTH MODEL ####
    elif provider is None:
        if "local-gptoss-unsloth" in model_name:
        # Extract url from model_name
        pattern = r"https?://"
        match = re.search(pattern, model_name)
        if match:
            start_index = match.start()
            url = model_name[start_index:]
        else:
            raise ValueError(f"Invalid URL in model name: {model_name}")
        client = openai.OpenAI(                                                                            # ADDED THIS LINE                
            api_key="filler",                                                                                  # ADDED THIS LINE        
            base_url=url
            )                                                                                    # ADDED THIS LINE
        if structured_output:                                                                        # ADDED THIS LINE              
            client = instructor.from_openai(                                                             # ADDED THIS LINE      
                client,                                                                                    # ADDED THIS LINE    
                mode=instructor.Mode.JSON,                                                             # ADDED THIS LINE                
            )      
            
    elif provider == "fugu":
        client = openai.OpenAI(
            api_key=os.environ["FUGU_API_KEY"],
            base_url=os.environ["FUGU_BASE_URL"],
            timeout=TIMEOUT,
        )
        if structured_output:
            raise ValueError("Fugu does not support structured output.")
    elif provider == "openrouter":
        client = openai.OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            timeout=TIMEOUT,  # 10 minutes
        )
        if structured_output:
            client = instructor.from_openai(client, mode=instructor.Mode.MD_JSON)
    else:
        raise ValueError(f"Model {model_name} not supported.")

    return client, model_name


def get_async_client_llm(
    model_name: str, structured_output: bool = False
) -> Tuple[Any, str]:
    """Get the async client and model for the given model name.

    Args:
        model_name (str): The name of the model to get the client.

    Raises:
        ValueError: If the model is not supported.

    Returns:
        The async client and model for the given model name.
    """
    # print(f"Getting async client for model {model_name}")
    provider = get_provider(model_name)

    if provider == "anthropic":
        client = anthropic.AsyncAnthropic(timeout=TIMEOUT)
        if structured_output:
            client = instructor.from_anthropic(
                client, mode=instructor.mode.Mode.ANTHROPIC_JSON
            )
    elif provider == "bedrock":
        client = anthropic.AsyncAnthropicBedrock(
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_region=os.getenv("AWS_REGION_NAME"),
            timeout=TIMEOUT,
        )
        if structured_output:
            client = instructor.from_anthropic(
                client, mode=instructor.mode.Mode.ANTHROPIC_JSON
            )
    elif provider == "openai":
        client = openai.AsyncOpenAI()
        if structured_output:
            client = instructor.from_openai(client, mode=instructor.Mode.TOOLS_STRICT)
    elif model_name.startswith("azure-"):
        # get rid of the azure- prefix
        model_name = model_name.split("azure-")[-1]
        client = openai.AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_API_ENDPOINT"),
            timeout=TIMEOUT,
        )
        if structured_output:
            client = instructor.from_openai(client, mode=instructor.Mode.TOOLS_STRICT)
    elif provider == "deepseek":
        client = openai.AsyncOpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
            timeout=TIMEOUT,
        )
        if structured_output:
            client = instructor.from_openai(client, mode=instructor.Mode.MD_JSON)
    elif provider == "google":
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        if structured_output:
            raise ValueError("Gemini does not support structured output.")
    elif provider == "fugu":
        client = openai.AsyncOpenAI(
            api_key=os.environ["FUGU_API_KEY"],
            base_url=os.environ["FUGU_BASE_URL"],
            timeout=60000,
        )
        if structured_output:
            raise ValueError("Fugu does not support structured output.")
    elif provider == "openrouter":
        client = openai.AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            timeout=TIMEOUT,
        )
        if structured_output:
            client = instructor.from_openai(client, mode=instructor.Mode.MD_JSON)
    else:
        raise ValueError(f"Model {model_name} not supported.")

    return client, model_name
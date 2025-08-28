import logging
from pathlib import Path
from typing import Any

import backoff
import numpy as np
from openai import APIError, APITimeoutError, OpenAI, RateLimitError

from agent_studio.config.config import Config
from agent_studio.llm.base_model import BaseModel
from agent_studio.llm.utils import openai_encode_image
from agent_studio.utils.types import MessageList

config = Config()
logger = logging.getLogger(__name__)


class OpenAIProvider(BaseModel):
    name = "openai"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.client = OpenAI(api_key=config.openai_api_key)

    def _format_messages(
        self,
        raw_messages: MessageList,
    ) -> list[dict[str, Any]]:
        """
        Composes the messages to be sent to the model.
        """
        model_message: list[dict[str, Any]] = []
        past_role = None
        for msg in raw_messages:
            if isinstance(msg.content, np.ndarray) or isinstance(msg.content, Path):
                content: dict = {
                    "type": "image_url",
                    "image_url": {"url": openai_encode_image(msg.content)},
                }
            elif isinstance(msg.content, str):
                content = {"type": "text", "text": msg.content}
            current_role = msg.role
            if past_role != current_role:
                model_message.append(
                    {
                        "role": current_role,
                        "content": [content],
                    }
                )
                past_role = current_role
            else:
                model_message[-1]["content"].append(content)
        return model_message

    def generate_response(
        self, messages: MessageList, **kwargs
    ) -> tuple[str, dict[str, Any]]:
        """Creates a chat completion using the OpenAI API."""

        model_name = kwargs.get("model", None)
        if model_name is None:
            raise ValueError("Model name is not set")
        temperature = kwargs.get("temperature", config.temperature)
        max_tokens = kwargs.get("max_tokens", config.max_tokens)

        # # HACKING!
        # if "variety of good ways" in str(messages):
        #     ret_str = 
        #     return ret_str, {}
        # # /HACKING!


        # Start by checking for the response in the cache.
        cache_ret = self._load_from_cache(model_name, self._hash_input(messages))
        if cache_ret is not None:
            logger.info("Found response in cache.")
            logger.info(f"Returning response:\n{cache_ret}")
            return cache_ret, {}  # TODO: for now we don't have any info to return
        # Else, generate a new response.
        model_message = self._format_messages(raw_messages=messages)
        # # Viz
        # for mm in model_message:
        #     for mmc in mm['content']:
        #         if 'image' not in mmc['type']:
        #             print(mmc)
        # import ipdb; ipdb.set_trace()
        # #
        logger.info(f"Creating chat completion with model {model_name}.")

        @backoff.on_exception(
            backoff.constant,
            (APIError, RateLimitError, APITimeoutError),
            max_tries=config.max_retries,
            interval=10,
        )
        def _generate_response_with_retry() -> tuple[str, dict[str, int]]:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=model_message,
                temperature=temperature,
                seed=config.seed,
                max_tokens=max_tokens,
            )

            if response is None:
                logger.error("Failed to get a response from OpenAI. Try again.")

            response_message = response.choices[0].message.content
            if response.usage is None:
                info = {}
                logger.warn("Failed to get usage information from OpenAI.")
            else:
                info = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "system_fingerprint": response.system_fingerprint,
                }

            logger.info(f"\nReceived response:\n{response_message}\nInfo:\n{info}")
            self._save_to_cache(
                model_name, self._hash_input(messages), response_message
            )
            return response_message, info

        return _generate_response_with_retry()

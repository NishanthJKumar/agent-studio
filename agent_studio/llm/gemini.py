import logging
from pathlib import Path
from typing import Any

import backoff
import google.generativeai as genai
import numpy as np

# Magic import, add following import to fix bug
# https://github.com/google/generative-ai-python/issues/178
import PIL.PngImagePlugin
from google.generativeai.types import GenerationConfig
from PIL import Image

from agent_studio.config.config import Config
from agent_studio.llm.base_model import BaseModel
from agent_studio.utils.types import MessageList

# Run this to pass mypy checker
PIL.PngImagePlugin


config = Config()
logger = logging.getLogger(__name__)


class GeminiProvider(BaseModel):
    name = "gemini"

    def __init__(self, seed: int, **kwargs) -> None:
        super().__init__(seed=seed)
        genai.configure(api_key=config.gemini_api_key)

    def _format_messages(self, raw_messages: MessageList):
        contents = []
        for msg in raw_messages:
            role = msg.role
            if role == "system":
                role = "user"
            if role == "assistant":
                role = "model"

            parts = []
            if isinstance(msg.content, str):
                parts.append(msg.content)
            elif isinstance(msg.content, np.ndarray):
                parts.append(Image.fromarray(msg.content).convert("RGB"))
            elif isinstance(msg.content, Path):
                parts.append(Image.open(msg.content).convert("RGB"))
            else:
                raise TypeError(f"Unknown message type: {type(msg.content)}")

            contents.append({"role": role, "parts": parts})
        return contents

    def generate_response(
        self, messages: MessageList, **kwargs
    ) -> tuple[str, dict[str, Any]]:
        """Creates a chat completion using the Gemini API."""
        model_name = kwargs.get("model", None)
        if model_name is None:
            raise ValueError("Model name is not set")
        temperature=kwargs.get("temperature", config.temperature)
        # Start by checking for the response in the cache.
        cache_ret = self._load_from_cache(model_name, self._hash_input(messages, temperature))
        if cache_ret is not None:
            logger.info("Found response in cache.")
            return cache_ret, {}  # TODO: for now we don't have any info to return
        # Else, generate a new response.
        model_message = self._format_messages(raw_messages=messages)
        if model_name is not None:
            model = genai.GenerativeModel(model_name)
        else:
            raise ValueError("Model name is required for GeminiProvider.")
        logger.info(f"Creating chat completion with model {model_name}.")

        generation_config = GenerationConfig(
            temperature=temperature,
            # top_p=kwargs.get("top_p", config.max_tokens),
            top_k=kwargs.get("top_k", config.top_k),
            candidate_count=1,
            # max_output_tokens=kwargs.get("max_tokens", config.max_tokens),
        )

        @backoff.on_exception(
            backoff.constant,
            genai.types.IncompleteIterationError,
            max_tries=config.max_retries,
            interval=10,
        )
        def _generate_response_with_retry() -> tuple[str, dict[str, int]]:
            response = model.generate_content(
                contents=model_message, generation_config=generation_config
            )
            token_count = model.count_tokens(model_message)
            info = {
                "total_tokens": token_count.total_tokens,
            }
            try:
                message = response.text
            except ValueError as e:
                message = ""
                logger.error(f"Failed to generate response: {e}")

            logger.info(f"\nReceived response:\n{message}\nInfo:\n{info}")
            self._save_to_cache(model_name, self._hash_input(messages, temperature), message)
            return message, info

        return _generate_response_with_retry()

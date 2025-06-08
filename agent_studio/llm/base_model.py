import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from agent_studio.config.config import Config
from agent_studio.utils.types import MessageList

config = Config()


class BaseModel:
    """Base class for models."""

    name: str = "base"

    def __init__(self) -> None:
        Path(config.pretrained_model_cache_dir).mkdir(parents=True, exist_ok=True)

    def _format_messages(
        self,
        raw_messages: MessageList,
    ) -> Any:
        raise NotImplementedError

    def _hash_input(self, messages: MessageList) -> str:
        """Create a hash for the input messages."""
        hasher = hashlib.sha256()
        for message in messages:
            hasher.update(str(message).encode("utf-8"))
        return hasher.hexdigest()

    def _load_from_cache(
        self, specific_model_name: str, hash_str: str
    ) -> Optional[str]:
        """Load from cache."""
        cache_path = (
            Path(config.pretrained_model_cache_dir) / specific_model_name / hash_str
        )
        if config.use_pretrained_model_cache and (cache_path).exists():
            with open(cache_path, "r") as f:
                return json.load(f)
        return None

    def _save_to_cache(
        self, specific_model_name: str, hash_str: str, response: str
    ) -> None:
        """Save to cache."""
        cache_path = Path(config.pretrained_model_cache_dir) / specific_model_name
        cache_path.mkdir(parents=True, exist_ok=True)
        if config.use_pretrained_model_cache:
            with open(cache_path / hash_str, "w") as f:
                json.dump(response, f)

    def generate_response(
        self, messages: MessageList, **kwargs
    ) -> tuple[str, dict[str, Any]]:
        """Generate a response given messages."""
        raise NotImplementedError

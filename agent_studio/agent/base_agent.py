import copy
import datetime
import logging
import time
from pathlib import Path

import numpy as np

from agent_studio.llm import ModelManager
from agent_studio.llm.utils import extract_from_response
from agent_studio.utils.runtime import PythonRuntime, RemotePythonRuntime
from agent_studio.utils.types import MessageList, StepInfo, TaskConfig

logger = logging.getLogger(__name__)


RUNTIME_INIT_CODE = """
from agent_studio.envs.desktop_env import Keyboard, Mouse


keyboard = Keyboard()
mouse = Mouse()
"""


class BaseAgent:
    """Base class for agents."""

    name: str = "base"

    def __init__(
        self,
        model: str,
        remote: bool,
        runtime_server_addr: str,
        runtime_server_port: int,
        results_dir: Path,
    ) -> None:
        """Initialize with model, prompt template, and initilization code."""
        model_manager = ModelManager()
        self.model = model_manager.get_model(model)
        self.remote = remote
        self.runtime_server_addr = runtime_server_addr
        self.runtime_server_port = runtime_server_port
        self.runtime: PythonRuntime | RemotePythonRuntime
        self.runtime_init_code: str = RUNTIME_INIT_CODE.strip()
        self.results_dir: Path = results_dir

        if self.remote:
            self.runtime = RemotePythonRuntime(
                env_server_addr=self.runtime_server_addr,
                env_server_port=self.runtime_server_port,
            )
        else:
            self.runtime = PythonRuntime()

        self.task_config: TaskConfig
        self.instruction: str
        self.trajectory: list[StepInfo]
        self.obs: np.ndarray | None = None
        self.step_info: StepInfo | None
        self.total_tokens: int

    def reset(self, task_config: TaskConfig) -> None:
        """Reset the agent's state with a new task configuration."""
        self.task_config = task_config
        self.instruction = task_config.instruction
        self.trajectory = []
        self.obs = None
        self.step_info: StepInfo | None = None
        self.total_tokens = 0

        self.runtime.reset()
        self.runtime(self.runtime_init_code)

    def generate_action(self, obs: np.ndarray | None, model_name: str) -> StepInfo:
        """Generate an action based on the observation."""
        self.obs = obs
        prompt = self.action_prompt
        assert prompt is not None, "Invalid prompt"
        logger.debug(f"Prompt: {prompt}")

        response, info = self.model.generate_response(messages=prompt, model=model_name)
        logger.debug(f"Response: {response}")
        assert response is not None, "Failed to generate response."
        self.total_tokens += info.get("total_tokens", 0)
        action = extract_from_response(response).strip()

        # Logging model outputs.
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output_{timestamp}.txt"
        log_dir = self.results_dir / "model_query_logs"
        log_dir.mkdir(exist_ok=True)
        filename = log_dir / filename
        with open(filename, "w") as file:
            file.write("Prompt:\n")
            for i in range(len(prompt)):
                file.write(f"Message {i}:\n")
                file.write(f"Role: {prompt[i].role}\n")
                file.write(f"Content: {prompt[i].content}\n\n")
            file.write("Response:\n")
            file.write(response + "\n")

        return StepInfo(
            obs=obs,
            prompt=prompt,
            response=response,
            action=action,
            info=info,
            result={},
            timestamp=0.0,
        )

    def step_action(
        self, failure_msg: str | None, step_info: StepInfo
    ) -> tuple[dict, bool]:
        """Execute the code if confirmed and record the result.
        If failure_msg is not None, the action is cancelled.
        """
        result = {}
        done = False
        if not failure_msg:
            code_clean = step_info.action
            if code_clean.endswith("exit()"):
                code = code_clean[: -len("exit()")].strip()
            else:
                code = code_clean
            logger.debug(f"Code to execute:\n{code}\n")
            result = self.runtime(code)
            # TODO: there might be other conditions to check for.
            if len(result.keys()) == 0:
                done = True
        else:
            result["force_stop_reason"] = failure_msg
            done = True

        step_info.result = copy.deepcopy(result)
        step_info.timestamp = time.time()
        self.trajectory.append(step_info)
        logger.info(f"Output: {result}")
        return result, done

    @property
    def action_prompt(self) -> MessageList:
        """Construct the action prompt."""
        raise NotImplementedError

    def get_token_count(self) -> int:
        """Return the total number of tokens used."""
        return self.total_tokens

    def close(self) -> None:
        """Close the runtime if it is open."""
        if self.runtime is not None:
            self.runtime.close()

import copy
import datetime
import logging
import time
from pathlib import Path

import numpy as np

from agent_studio.llm import ModelManager
from agent_studio.llm.utils import extract_from_response
from agent_studio.utils.runtime import PythonRuntime, RemotePythonRuntime
from agent_studio.utils.types import Message, MessageList, StepInfo, TaskConfig

logger = logging.getLogger(__name__)


RUNTIME_INIT_CODE = """
import subprocess
import time
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
        restrict_to_one_step: bool,
        prompt_approach: str = "naive",
        model_server: str = None,
        summarization_prompt_approach: str = "naive",
    ) -> None:
        """Initialize with model, prompt template, and initilization code."""
        model_manager = ModelManager()
        self.model = model_manager.get_model(model, model_server=model_server)
        self.remote = remote
        self.runtime_server_addr = runtime_server_addr
        self.runtime_server_port = runtime_server_port
        self.runtime: PythonRuntime | RemotePythonRuntime
        self.runtime_init_code: str = RUNTIME_INIT_CODE.strip()
        self.results_dir: Path = results_dir
        self.restrict_to_one_step = restrict_to_one_step

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
        with open(
            f"agent_studio/agent/prompts/{summarization_prompt_approach}_summary_prompt.txt", "r"
        ) as file:
            self._summarization_prompt = file.read()
        self.prev_attempt_summaries = []

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
        # logger.debug(f"Prompt: {prompt}")
        response, info = self.model.generate_response(messages=prompt, model=model_name)
        # logger.debug(f"Response: {response}")
        assert response is not None, "Failed to generate response."
        self.total_tokens += info.get("total_tokens", 0)
        action = extract_from_response(response).strip()
        if action == "":
            logger.info("Output response didn't contain action; trying again!")
            new_message = Message(
                role="user",
                content=f"ERROR! You just output '''{response}'''. However, this "
                "did not contain a valid ```python``` code block. Please "
                "try again and ensure your response contains a valid "
                "```python``` codeblock.",
            )
            error_recovery_prompt = prompt[:-1] + [new_message] + [prompt[-1]]
            response, info = self.model.generate_response(
                messages=error_recovery_prompt, model=model_name
            )
            self.total_tokens += info.get("total_tokens", 0)
            action = extract_from_response(response).strip()

        unexecuted_code = ""
        if self.restrict_to_one_step:
            # Truncate action if it contains keyboard or mouse commands.
            if "keyboard." in action or "mouse." in action:
                truncated_code = ""
                # Split the code into lines
                code_lines = action.splitlines()
                # Find the first line containing "keyboard." or "mouse."
                for line in code_lines:
                    truncated_code += line + "\n"
                    if "keyboard." in line or "mouse." in line:
                        break
                logger.info(f"Truncating code from: {action}\n to: {truncated_code}")
                unexecuted_code = action[len(truncated_code) :]
                action = truncated_code

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
            unexecuted_code=unexecuted_code,
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
        exit_in_code = False
        if not failure_msg:
            code_clean = step_info.action
            if code_clean.endswith("exit()"):
                code = code_clean[: -len("exit()")].strip()
                exit_in_code = True
            else:
                code = code_clean
            logger.info(f"Code to execute:\n{code}\n")
            if len(code) > 0:
                result = self.runtime(code)
            else:
                result = {}
            # TODO: there might be other conditions to check for.
            if len(result.keys()) == 0 and exit_in_code:
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

    def construct_traj_summary(self, model_name: str, succeeded: bool, test_feedback: str) -> str:
        """Construct a summary of the trajectory.
        
        Useful for REFLEXION-style learning and
        exploration.
        """
        summarization_prompt = self.traj_summary_prompt(succeeded, test_feedback)
        traj_summary, _ = self.model.generate_response(messages=summarization_prompt, model=model_name)
        return traj_summary

    def traj_summary_prompt(self, succeeded: bool, test_feedback: str) -> MessageList:
        """Construct the trajectory summary prompt."""
        messages: MessageList = []
        messages.append(Message(role="system", content=self._summarization_prompt))
        messages.append(
            Message(role="user", content=f"The task instruction: {self.instruction}")
        )

        for i, step in enumerate(self.trajectory):
            messages.append(
                Message(
                    role="assistant",
                    content=f"##Step number: {i}.\n"
                    f"##Executed Action: ```python\n{step.action}\n```\n"
                    f"##Python Execution Result of Action:\n{step.result}"
                ))
            messages.append(
                Message(
                    role="user",
                    content = step.obs
                )
            )

        # Add in final message about outcome.
        succeeded_str = "Succeeded." if succeeded else f"Failed. {test_feedback}"
        messages.append(Message(role="user", content=f"##Outcome: {succeeded_str}\n"))

        return messages
"""An agent inspired by Hierarchical Planning approaches and the open-source
SOTA Browser-Use (https://browser-use.com/).
Overall, this is simply a more structured version of the direct prompting
agent that uses data structures/classes to store different elements of the
state and agent's plan instead of just doing everything implicitly within
the prompt and the agent's response to the prompt"""

import copy
import datetime
import logging
import time
from pathlib import Path

import numpy as np

from agent_studio.agent.base_agent import BaseAgent
from agent_studio.llm import ModelManager
from agent_studio.llm.utils import (
    extract_from_response,
    structured_json_extract_from_response,
)
from agent_studio.utils.runtime import PythonRuntime, RemotePythonRuntime
from agent_studio.utils.types import (
    Message,
    MessageList,
    StructuredStepInfo,
    TaskConfig,
)

logger = logging.getLogger(__name__)


RUNTIME_INIT_CODE = """
from agent_studio.envs.desktop_env import Keyboard, Mouse


keyboard = Keyboard()
mouse = Mouse()
"""


class StructuredPlanningAgent(BaseAgent):
    """Class for a structured (hierarchical) planning agent."""

    name: str = "structured_planning"

    def __init__(
        self,
        model: str,
        remote: bool,
        runtime_server_addr: str,
        runtime_server_port: int,
        results_dir: Path,
        restrict_to_one_step: bool,
        prompt_approach: str = "naive",
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
        self.trajectory: list[StructuredStepInfo]
        self.obs: np.ndarray | None = None
        self.step_info: StructuredStepInfo | None
        self.total_tokens: int
        with open(
            f"agent_studio/agent/prompts/{prompt_approach}_system_prompt.txt", "r"
        ) as file:
            self._system_prompt = file.read()

    def reset(self, task_config: TaskConfig) -> None:
        """Reset the agent's state with a new task configuration."""
        self.task_config = task_config
        self.instruction = task_config.instruction
        self.trajectory = []
        self.obs = None
        self.step_info: StructuredStepInfo | None = None
        self.total_tokens = 0

        self.runtime.reset()
        self.runtime(self.runtime_init_code)

    def generate_action(
        self, obs: np.ndarray | None, model_name: str
    ) -> StructuredStepInfo:
        """Generate an action based on the observation."""
        self.obs = obs
        prompt = self.action_prompt
        assert prompt is not None, "Invalid prompt"
        response, info = self.model.generate_response(messages=prompt, model=model_name)
        assert response is not None, "Failed to generate response."
        self.total_tokens += info.get("total_tokens", 0)
        json_output = structured_json_extract_from_response(response)
        # TODO: make the below parsing safer + less brittle via requerying.
        action = extract_from_response(json_output["action"]).strip()
        new_high_level_plan = json_output["high_level_plan"]
        if new_high_level_plan == "No change." and len(self.trajectory) > 0:
            new_high_level_plan = self.trajectory[-1].current_high_level_plan
        curr_state_analysis = json_output["state_analysis"]
        prev_goal_achieved = json_output["previous_goal_achieved"]
        next_action_result = json_output["intended_action_result"]

        # if action == "":
        #     logger.info("Output response didn't contain action; trying again!")
        #     new_message = Message(
        #         role="user",
        #         content=f"ERROR! You just output '''{response}'''. However, this "
        #         "did not contain a valid ```python``` code block. Please "
        #         "try again and ensure your response contains a valid "
        #         "```python``` codeblock.",
        #     )
        #     error_recovery_prompt = prompt[:-1] + [new_message] + [prompt[-1]]
        #     response, info = self.model.generate_response(
        #         messages=error_recovery_prompt, model=model_name
        #     )
        #     self.total_tokens += info.get("total_tokens", 0)
        #     action = extract_from_response(response).strip()

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

        return StructuredStepInfo(
            obs=obs,
            prev_expected_result_achieved=prev_goal_achieved,
            prompt=prompt,
            current_high_level_plan=new_high_level_plan,
            action=action,
            current_scene_description=curr_state_analysis,
            next_expected_result=next_action_result,
            result={},
            info={},
            timestamp=0.0,
        )

    def step_action(
        self, failure_msg: str | None, step_info: StructuredStepInfo
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
        messages: MessageList = []
        messages.append(Message(role="system", content=self._system_prompt))
        messages.append(
            Message(role="user", content=f"The task instruction: {self.instruction}")
        )

        for i, step in enumerate(self.trajectory):
            high_level_plan_str = "\n".join(
                f"{i}. {hls}" for i, hls in enumerate(step.current_high_level_plan)
            )
            messages.append(
                Message(
                    role="assistant",
                    content=f"##Step number: {i}.\n"
                    f"##State Analysis: {step.current_scene_description}\n"
                    f"##High-Level Plan: {high_level_plan_str}"
                    "Previous Action's Goal Achieved?: "
                    f"{step.prev_expected_result_achieved}\n"
                    f"##Executed Action: ```python\n{step.action}\n```\n"
                    "##Intended Result of Action Execution: "
                    f"{step.next_expected_result}\n"
                    f"##Python Execution Result of Action:\n{step.result}",
                )
            )

        if self.obs is not None:
            messages.append(Message(role="user", content=self.obs))

        return messages

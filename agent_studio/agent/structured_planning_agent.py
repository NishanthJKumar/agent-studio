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
from agent_studio.llm.utils import (
    extract_from_response,
    structured_json_extract_from_response,
)
from agent_studio.utils.types import (
    Message,
    MessageList,
    StructuredStepInfo,
    TaskConfig,
)

logger = logging.getLogger(__name__)


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
        model_server: str = None,
        summarization_prompt_approach: str = "naive",
        extra_args: dict = {}
    ) -> None:
        """Initialize with model, prompt template, and initilization code."""
        super().__init__(
            model=model,
            remote=remote,
            runtime_server_addr=runtime_server_addr,
            runtime_server_port=runtime_server_port,
            results_dir=results_dir,
            restrict_to_one_step=restrict_to_one_step,
            prompt_approach=prompt_approach,
            model_server=model_server,
        )

        # Override the following variables
        self.trajectory: list[StructuredStepInfo]
        self.step_info: StructuredStepInfo | None
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

        # Reprompting loop for correct formatting.
        action = None
        new_high_level_plan = None
        curr_state_analysis = None
        prev_goal_achieved = None
        next_action_result = None
        for _ in range(5):
            try:
                json_output = structured_json_extract_from_response(response)
                action = extract_from_response(json_output["action"]).strip()
                if action == "":
                    raise KeyError("No action with ```python``` codeblock found.")
                new_high_level_plan = json_output["high_level_plan"]
                if new_high_level_plan == "No change." and len(self.trajectory) > 0:
                    new_high_level_plan = self.trajectory[-1].current_high_level_plan
                curr_state_analysis = json_output["state_analysis"]
                prev_goal_achieved = json_output["previous_goal_achieved"]
                next_action_result = json_output["intended_action_result"]
            except Exception as e:
                logger.info(f"Output response badly formatted! Error: {e}")
                new_message = Message(
                    role="user",
                    content=f"ERROR! You just output '''{response}'''. However, this "
                    "was badly formatted. In particular, it lead to the "
                    "following error "
                    f"{e}. Please try again and fix this error; "
                    "ensure your response contains "
                    "valid JSON with all the fields requested above. Also "
                    "ensure that the action field contains exactly a single string "
                    "that has a valid ```python``` codeblock within it. "
                    "Finally, ensure that this codeblock has `\n` characters "
                    "and not direct newline characters that you insert.",
                )
                error_recovery_prompt = prompt + [new_message]
                response, info = self.model.generate_response(
                    messages=error_recovery_prompt, model=model_name
                )
                self.total_tokens += info.get("total_tokens", 0)

            if (
                action is not None
                and len(action) > 0
                and new_high_level_plan is not None
                and curr_state_analysis is not None
                and prev_goal_achieved is not None
            ):
                break

        assert action is not None
        assert new_high_level_plan is not None
        assert curr_state_analysis is not None
        assert prev_goal_achieved is not None

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
        return super().step_action(failure_msg, step_info)

    @property
    def action_prompt(self) -> MessageList:
        messages: MessageList = []
        messages.append(Message(role="system", content=self._system_prompt))        
        messages.append(
            Message(role="user", content=f"The task instruction: {self.instruction}")
        )
        if len(self.prev_attempt_summaries) > 0:
            messages.append(Message(role="user", content="To help you with this task, here are summaries of your previous attempts. Please use these to inform your planning and decision-making: try to improve on past failures and build on past successes!"))
            for i, attempt_summary in enumerate(self.prev_attempt_summaries):
                messages.append(Message(role="user", content=f"Attempt {i}: {attempt_summary}"))

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

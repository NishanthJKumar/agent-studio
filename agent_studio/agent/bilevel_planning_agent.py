"""An agent that plans in two stages: first generating high-level ''task plans''
and then refining these into generate low-level actions."""

from typing import Optional
import copy
import datetime
import logging
import time
from pathlib import Path
import json
import os
import cv2

import numpy as np

from agent_studio.agent.structured_planning_agent import StructuredPlanningAgent
from agent_studio.llm import ModelManager
from agent_studio.llm.utils import (
    extract_from_response,
    structured_json_extract_from_response,
    parse_strategies
)
from agent_studio.utils.types import (
    Message,
    MessageList,
    StructuredStepInfo,
    TaskConfig,
)

logger = logging.getLogger(__name__)


class BilevelPlanningAgent(StructuredPlanningAgent):
    """A bilevel planning agent."""

    name: str = "bilevel_planning"

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
        self.prev_task_config: Optional[TaskConfig] = None
        self.episode_idx: int = 0
        self.high_level_plan_candidates: list[str] = []
        self.extra_args = extra_args
        assert "scoring_approach" in self.extra_args, "Must specify scoring_approach."
        self.critic_model = None
        if self.extra_args["scoring_approach"] == "critic":
            model_manager = ModelManager()
            self.critic_model = model_manager.get_model(self.extra_args["scoring_model_name"], model_server=model_server)


    def reset(self, task_config: TaskConfig) -> None:
        """Reset the agent's state with a new task configuration."""
        super().ƒ(task_config)
        if self.task_config != self.prev_task_config:
            self.curr_high_level_plan_idx = 0
            self.high_level_plan_candidates = []
            self.prev_task_config = copy.deepcopy(self.task_config)
        else:
            self.episode_idx += 1
            logger.info(f"\n\nCurrent high-level plan: {self.high_level_plan_candidates[self.episode_idx % len(self.high_level_plan_candidates)]}\n\n")


    def generate_new_high_level_plan_candidates(self, obs: np.ndarray | None, planning_model_name: str, scoring_approach: str, scoring_model_name: str) -> None:
        """Generate new high-level plan candidates."""
        with open(
            f"agent_studio/agent/prompts/diversity_hint_prompt.txt", "r"
        ) as file:
            diversity_prompt = file.read()
        messages: MessageList = []
        messages.append(Message(role="system", content=diversity_prompt))
        messages.append(
            Message(role="user", content=f"The task instruction: {self.task_config.instruction}")
        )
        if self.obs is not None:
            messages.append(Message(role="user", content=obs))
        hint_response, _ = self.model.generate_response(messages=messages, model=planning_model_name)
        self.high_level_plan_candidates = parse_strategies(hint_response)
        assert len(self.high_level_plan_candidates) > 0, "No high-level plan candidates generated."
        logger.info(f"Got new task: generated {len(self.high_level_plan_candidates)} high-level plan candidates.")
        logger.info(f"Scoring plans with strategy {scoring_approach}.")
        plans_to_scores = {}
        for i, curr_high_level_plan in enumerate(self.high_level_plan_candidates):
            curr_score = self.score_high_level_plan(curr_high_level_plan, scoring_model_name, scoring_approach)
            plans_to_scores[curr_high_level_plan] = curr_score
            logger.info(f"Scored plan {i}: {curr_score}")
        self.high_level_plan_candidates = [k for k, v in sorted(plans_to_scores.items(), key=lambda item: item[1], reverse=True)]


    def score_high_level_plan(self, curr_high_level_plan: str, model_name: str, scoring_approach: str) -> float:
        """Score a high-level plan.
        
        Right now, this is hardcoded to a specific scoring function and scheme. But in
        the future, we can make this more flexible.
        """
        if scoring_approach == "uniform":
            return 0.0
        elif scoring_approach == "critic":
            HINT_PRED_PROMPT = ("You are an expert-level predictor of whether or not a particular strategy will work for various computer use tasks."
                "You will be provided with a task instruction (natural language string), potentially an image of the initial state of the environment before task execution, and an agent's strategy to complete the task."
                "Your job is to determine whether this strategy will succeed at accomplishing the task or not."
                "If you think it will succeed, output '[PREDICTED OUTCOME] Success.'; otherwise indicate failure. You can also output an explanation of your approach."
            )
            messages: MessageList = []
            messages.append(Message(role="system", content=HINT_PRED_PROMPT))
            messages.append(Message(role="user",
                    content=f"Task instruction: {self.task_config.instruction}\nAgent Plan Strategy: {curr_high_level_plan}"
                )
            )
            if self.obs is not None:
                messages.append(Message(role="user", content=self.obs))
            response, _ = self.critic_model.generate_response(messages=messages, model=model_name)
            # NOTE: very simple scoring function for now; can be improved/changed later!
            if "success" in response.lower():
                return 1.0
            else:
                return 0.0
        else:
            raise ValueError(f"Unknown scoring approach: {scoring_approach}")


    def generate_action(
        self, obs: np.ndarray | None, model_name: str
    ) -> StructuredStepInfo:
        """Generate an action based on the observation."""
        if len(self.high_level_plan_candidates) == 0:
            self.generate_new_high_level_plan_candidates(obs, model_name, self.extra_args["scoring_approach"], self.extra_args["scoring_model_name"])
        self.obs = obs
        prompt = self.action_prompt
        assert prompt is not None, "Invalid prompt"
        response, info = self.model.generate_response(messages=prompt, model=model_name)
        assert response is not None, "Failed to generate response."
        self.total_tokens += info.get("total_tokens", 0)
        curr_high_level_plan = self.high_level_plan_candidates[self.episode_idx % len(self.high_level_plan_candidates)]

        # Reprompting loop for correct formatting.
        action = None
        curr_state_analysis = None
        prev_goal_achieved = None
        next_action_result = None
        for _ in range(5):
            try:
                json_output = structured_json_extract_from_response(response)
                action = extract_from_response(json_output["action"]).strip()
                if action == "":
                    raise KeyError("No action with ```python``` codeblock found.")
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
                and curr_state_analysis is not None
                and prev_goal_achieved is not None
            ):
                break

        assert action is not None
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
            current_high_level_plan=curr_high_level_plan,
            action=action,
            current_scene_description=curr_state_analysis,
            next_expected_result=next_action_result,
            result={},
            info={},
            timestamp=0.0,
        )


    @property
    def action_prompt(self) -> MessageList:
        messages: MessageList = []
        messages.append(Message(role="system", content=self._system_prompt))        
        messages.append(
            Message(role="user", content=f"The task instruction: {self.instruction}")
        )
        # The current high-level plan is the one that is the episode_idx modulo
        # the number of high-level plans available.
        curr_high_level_plan = self.high_level_plan_candidates[self.episode_idx % len(self.high_level_plan_candidates)]
        messages.append(
            Message(role="user", content=f"#High-Level Plan to follow: {curr_high_level_plan}")
        )
        if len(self.prev_attempt_summaries) > 0:
            messages.append(Message(role="user", content="To help you with this task, here are summaries of your previous attempts. Please use these to inform your planning and decision-making: try to improve on past failures and build on past successes!"))
            for i, attempt_summary in enumerate(self.prev_attempt_summaries):
                messages.append(Message(role="user", content=f"Attempt {i}: {attempt_summary}"))

        for i, step in enumerate(self.trajectory):
            messages.append(
                Message(
                    role="assistant",
                    content=f"##Step number: {i}.\n"
                    f"##State Analysis: {step.current_scene_description}\n"
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


    def save_finetuning_data(self, outcome: bool, steps_taken: int, init_obs: np.ndarray | None, data_save_path: str = "finetuning_data") -> None:
        """Save finetuning data."""
        # Create directory if it doesn't exist
        save_dir = Path(data_save_path)
        save_dir.mkdir(exist_ok=True)
        image_dir = save_dir / "images"
        image_dir.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save image if provided
        init_obs_img_path = None
        if init_obs is not None:
            img_filename = f"{self.task_config.task_id}_img_{timestamp}.png"
            img_path = image_dir / img_filename
            cv2.imwrite(str(img_path), init_obs)
            init_obs_img_path = str(Path("images") / img_filename)

        curr_high_level_plan = self.high_level_plan_candidates[self.episode_idx % len(self.high_level_plan_candidates)]
        task_string = self.instruction
        data = {
        "task_string": task_string,
        "initial_image_path": init_obs_img_path,
        "hint_string": curr_high_level_plan,
        "outcome": outcome,
        "trajectory_metadata": {
            "steps_taken": steps_taken,
            }
        }
        # Generate unique filename
        filename = save_dir / f"{self.task_config.task_id}_traj_{timestamp}.json"
        # Save the JSON file
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)        
        logger.info(f"Saved finetuning data to {filename}")


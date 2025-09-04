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
from agent_studio.utils.json_utils import read_json

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
        self.curr_high_level_plan: Optional[str] = None
        self.extra_args = extra_args
        assert "scoring_approach" in self.extra_args, "Must specify scoring_approach."
        self.critic_model = None
        if "critic" in self.extra_args["scoring_approach"]:
            model_manager = ModelManager()
            self.critic_model = model_manager.get_model(self.extra_args["scoring_model_name"], model_server=model_server)
        assert "plan_proposing_approach" in self.extra_args, "Must specify plan_proposing_approach."
        self.plan_proposing_approach: str = self.extra_args["plan_proposing_approach"]
        self.num_plan_examples_to_sample: int = self.extra_args.get("num_plan_examples_to_sample", 5)
        self.num_unique_plan_candidates: int = self.extra_args.get("num_unique_plan_candidates", 5)
        self.existing_plans_location: Optional[str] = self.extra_args.get("existing_plans_location", None)
        self.rng = np.random.default_rng(23)


    def reset(self, task_config: TaskConfig) -> None:
        """Reset the agent's state with a new task configuration."""
        super().reset(task_config)
        if self.task_config != self.prev_task_config:
            self.episode_idx = 0
            self.high_level_plan_candidates = []
            self.prev_task_config = copy.deepcopy(self.task_config)
        else:
            self.episode_idx += 1
            assert self.episode_idx < len(self.high_level_plan_candidates), "Not enough high-level plans available; shouldn't happen (3)!"
            self.curr_high_level_plan = self.high_level_plan_candidates[self.episode_idx]
            logger.info(f"\n\nCurrent high-level plan: {self.curr_high_level_plan}\n\n")


    def _generate_high_level_plan_candidates_from_scratch(self, obs: np.ndarray | None, planning_model_name: str) -> set[str]:
        # Start by generating initial candidate plans set.
        with open(
            f"agent_studio/agent/prompts/strategy_generation_prompt.txt", "r"
        ) as file:
            diversity_prompt = file.read()
        messages: MessageList = []
        diversity_prompt = diversity_prompt.format(task_instruction=self.task_config.instruction)
        messages.append(Message(role="system", content=diversity_prompt))
        if obs is not None:
            messages.append(Message(role="user", content=obs))
        logger.info(f"Got new task: generating plan candidates!")
        hint_response, _ = self.model.generate_response(messages=messages, model=planning_model_name, temperature=0.75)        
        return set(parse_strategies(hint_response))


    def _generate_high_level_plan_candidates_from_examples(self,
                                                        example_plans: list[str], 
                                                        obs: np.ndarray | None, 
                                                        planning_model_name: str, 
                                                        example_bootstrapping_approach: str) -> set[str]:
        """Generate new high-level plan candidates."""
        assert len(example_plans) > 0, "No high-level plan candidates available."
        if example_bootstrapping_approach == "diversity":
            prompt_hint_filepath = "agent_studio/agent/prompts/strategy_growth_differencing_hint_prompt.txt"
            temp = 0.75
        elif example_bootstrapping_approach == "similarity":
            prompt_hint_filepath = "agent_studio/agent/prompts/strategy_growth_similar_hint_prompt.txt"
            temp = 0.0
        else:
            raise ValueError(f"example_bootstrapping_approach {example_bootstrapping_approach} unknown.")
        with open(
            prompt_hint_filepath, "r"
        ) as file:
            growth_prompt = file.read()
        growth_prompt = growth_prompt.format(example_plans="\n\n".join(plan for plan in example_plans), task_instruction=self.task_config.instruction)
        messages: MessageList = []
        messages.append(Message(role="system", content=growth_prompt))
        if self.obs is not None:
            messages.append(Message(role="user", content=obs))
        hint_response, _ = self.model.generate_response(messages=messages, model=planning_model_name, temperature=temp)
        return set(parse_strategies(hint_response))


    def _score_and_order_plans_list(self, plans_list: list[str], scoring_approach: str, scoring_model_name: str) -> list[str]:
        plans_to_scores = {}
        for i, curr_high_level_plan in enumerate(plans_list):
            curr_score = self.score_high_level_plan(curr_high_level_plan, scoring_model_name, scoring_approach)
            plans_to_scores[curr_high_level_plan] = curr_score
            logger.info(f"Scored plan \n {curr_high_level_plan} \n: {curr_score}")
        new_plans_list = [k for k, v in sorted(plans_to_scores.items(), key=lambda item: item[1], reverse=True)]
        return new_plans_list


    def generate_additional_high_level_plans_from_examples(self, obs: np.ndarray | None, 
                                                            init_plan_candidates_set: set[str], 
                                                            example_plans_source_set: set[str], 
                                                            planning_model_name: str, 
                                                            scoring_approach: str,
                                                            plan_proposing_approach: str, 
                                                            scoring_model_name: str,
                                                            num_candidates_to_generate: str,) -> set[str]:
        """Use some initial plans to bootstrap generation of additional plans that are similar or different to these"""
        # Scale up and generate additional plans.
        if plan_proposing_approach == "diversity":
            while len(init_plan_candidates_set) < num_candidates_to_generate:
                sample_size = min(len(example_plans_source_set), self.num_plan_examples_to_sample)
                plan_examples = self.rng.choice(list(example_plans_source_set), size=sample_size, replace=False)
                new_plans_set = self._generate_high_level_plan_candidates_from_examples(plan_examples, obs, planning_model_name, "diversity")
                unique_new_plans_set = new_plans_set - init_plan_candidates_set - example_plans_source_set
                init_plan_candidates_set = init_plan_candidates_set | unique_new_plans_set
                example_plans_source_set = example_plans_source_set | unique_new_plans_set
                logger.info(f"Curr total plan candidates: {len(init_plan_candidates_set)}.")
        elif plan_proposing_approach == "top_score_similarity":
            while len(init_plan_candidates_set) < num_candidates_to_generate:
                new_plans_set = set()
                logger.info("SCORING AND ORDERING INITIAL PLAN CANDIDATES.")
                ordered_plans = self._score_and_order_plans_list(sorted(init_plan_candidates_set), scoring_approach, scoring_model_name)
                logger.info("DONE SCORING AND ORDERING INITIAL PLAN CANDIDATES.")
                # NOTE: we only use the top 5 plans for now; could change this in the future.
                logger.info("ORDERED INIT PLAN CANDIDATES:")
                for i, curr_high_level_plan in enumerate(ordered_plans[:5]):
                    logger.info(f"{i}: {curr_high_level_plan}\n")
                logger.info(f"\nEND ORDERED INIT PLANS\n")
                for i in range(self.num_plan_examples_to_sample):
                    logger.info(f"GENERATING PLANS SIMILAR TO PLAN {i}: {ordered_plans[i]}.")
                    new_plans_set |= self._generate_high_level_plan_candidates_from_examples([ordered_plans[i]], obs, planning_model_name, "similarity")
                    logger.info(f"DONE GENERATING PLANS SIMILAR TO PLAN {i}.")
                unique_new_plans_set = new_plans_set - init_plan_candidates_set - example_plans_source_set
                init_plan_candidates_set = init_plan_candidates_set | unique_new_plans_set
                example_plans_source_set = example_plans_source_set | unique_new_plans_set
                logger.info(f"Curr total plan candidates: {len(init_plan_candidates_set)}.")
        else:
            raise ValueError(f"plan_proposing_approach {plan_proposing_approach} unknown.")
        return init_plan_candidates_set


    def generate_high_level_plan_candidates(self, 
                                            obs: np.ndarray | None, 
                                            planning_model_name: str, 
                                            scoring_approach: str, 
                                            scoring_model_name: str) -> None:
        """Generate new high-level plan candidates."""
        self.rng = np.random.default_rng(23) # <- ensure determinism; can change later to vary seeds over runs.
        plan_candidates_set = set()
        example_plans_source_set = set()
        if self.existing_plans_location is None:
            plan_candidates_set = self._generate_high_level_plan_candidates_from_scratch(obs, planning_model_name)         
            example_plans_source_set = copy.deepcopy(plan_candidates_set)
        else:
            # TODO: need to potentially load from multiple folders for rounds beyond 2!
            loaded_plans = []
            # Load existing plans while taking care to do so from the correct task.
            assert self.task_config is not None, "Task config not set!"
            task_name = self.task_config.task_id
            assert task_name is not None, "Task name not set!"
            existing_plans_location = Path(self.existing_plans_location)
            assert existing_plans_location.is_dir(), "Existing plans location is not a directory!"
            for json_file in existing_plans_location.glob("*.json"):
                filename = json_file.stem
                curr_json_task_name = filename.split("_traj")[0]
                if curr_json_task_name != task_name:
                    continue
                data = read_json(json_file)
                plan = data["hint_string"]
                loaded_plans.append(plan)
            example_plans_source_set = set(loaded_plans)
            assert len(loaded_plans) == len(example_plans_source_set), "Duplicate plans found!"
            logger.info(f"Loaded {len(loaded_plans)} existing plans.")
            if self.plan_proposing_approach != "diversity":
                # Now, we need to generate an initial set of new plans to bootstrap the 
                # similarity generation process.
                plan_candidates_set = self.generate_additional_high_level_plans_from_examples(obs, 
                                        plan_candidates_set, 
                                        example_plans_source_set, 
                                        planning_model_name, 
                                        scoring_approach,
                                        "diversity",
                                        scoring_model_name,
                                        self.num_plan_examples_to_sample)
                example_plans_source_set = example_plans_source_set | plan_candidates_set


        logger.info(f"Currently have {len(plan_candidates_set)} high-level plan candidates. Need {self.num_unique_plan_candidates}.")

        # Come up with a set of candidates - this may or may not use the scoring model implicitly.
        plan_candidates_set = self.generate_additional_high_level_plans_from_examples(obs, 
                                    plan_candidates_set, 
                                    example_plans_source_set, 
                                    planning_model_name, 
                                    scoring_approach,
                                    self.plan_proposing_approach,
                                    scoring_model_name,
                                    self.num_unique_plan_candidates)

        # Score the plans to order them.
        self.high_level_plan_candidates = self._score_and_order_plans_list(sorted(plan_candidates_set), scoring_approach, scoring_model_name)
        assert self.episode_idx < len(self.high_level_plan_candidates), "Not enough high-level plans available; shouldn't happen (1)!"
        self.curr_high_level_plan = self.high_level_plan_candidates[self.episode_idx]
        if "critic" in scoring_approach:
            logger.info(f"RANKED PLANS:\n\n")
            for i, curr_high_level_plan in enumerate(self.high_level_plan_candidates[:5]):
                logger.info(f"{i}: {curr_high_level_plan}\n")
            logger.info(f"\nEND RANKED PLANS\n")


    def score_high_level_plan(self, curr_high_level_plan: str, model_name: str, scoring_approach: str) -> float:
        """Score a high-level plan.
        
        Right now, this is hardcoded to a specific scoring function and scheme. But in
        the future, we can make this more flexible.
        """
        if scoring_approach == "uniform":
            return 0.0
        elif "critic" in scoring_approach:
            messages: MessageList = []
            HINT_PRED_PROMT = ("You are an expert-level predictor of whether or not a particular strategy will work for various computer use tasks."
                "You will be provided with a task instruction (natural language string), potentially an image of the initial state of the environment before task execution, and an agent's strategy to complete the task."
                "Your job is to determine whether this strategy will succeed at accomplishing the task or not."
                f"You must answer in the following format exactly: 'Success.' or 'Failure.'. No extra words.")
            messages.append(Message(role="system", content=HINT_PRED_PROMT))
            if self.obs is not None:
                messages.append(Message(role="user", content=self.obs))
            messages.append(Message(role="user",
                    content=f"Task Instruction: {self.task_config.instruction}\nAgent Plan Strategy: {curr_high_level_plan}"
                )
            )
            response, extra_info = self.critic_model.generate_response(messages=messages, model=model_name)
            # NOTE: we hardcode the classes to be "success" and "failure" here; we'll have to change this if we
            # change classes. Might be worth adding a class mapping to the config/some kind of more general
            # mechanism.
            if scoring_approach == "critic_naive":
                if "success" in response.lower():
                    return 1.0
                else:
                    return 0.0
            elif scoring_approach == "critic_log_prob_mag":
                assert "logit_scores" in extra_info, "Logit scores not found in extra info."
                logit_scores = extra_info["logit_scores"]
                return logit_scores["Success."]
            elif scoring_approach == "critic_log_prob_diff":
                assert "logit_scores" in extra_info, "Logit scores not found in extra info."
                logit_scores = extra_info["logit_scores"]
                success_logit = logit_scores["Success."]
                failure_logit = logit_scores["Failure."]
                return success_logit - failure_logit              
            else:
                raise ValueError(f"Unknown scoring approach: {scoring_approach}")  
        else:
            raise ValueError(f"Unknown scoring approach: {scoring_approach}")


    def generate_action(
        self, obs: np.ndarray | None, model_name: str
    ) -> StructuredStepInfo:
        """Generate an action based on the observation."""
        self.obs = obs
        if len(self.high_level_plan_candidates) == 0:
            self.generate_high_level_plan_candidates(obs, model_name, self.extra_args["scoring_approach"], self.extra_args["scoring_model_name"])
            logger.info(f"\n\nCurr plan: {self.high_level_plan_candidates[self.episode_idx % len(self.high_level_plan_candidates)]}\n\n")        
        prompt = self.action_prompt
        assert prompt is not None, "Invalid prompt"
        response, info = self.model.generate_response(messages=prompt, model=model_name)
        assert response is not None, "Failed to generate response."
        self.total_tokens += info.get("total_tokens", 0)

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
        if "log_model_output" in self.extra_args and self.extra_args["log_model_output"]:
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
            current_high_level_plan=self.curr_high_level_plan,
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
            Message(role="user", content=f"#High-Level Plan to follow: {self.curr_high_level_plan}")
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

        assert self.episode_idx < len(self.high_level_plan_candidates), "Not enough high-level plans available; shouldn't happen (2)!"
        curr_high_level_plan = self.high_level_plan_candidates[self.episode_idx % len(self.high_level_plan_candidates)]
        task_string = self.instruction
        data = {
        "task_string": task_string,
        "initial_image_path": init_obs_img_path,
        "hint_string": self.curr_high_level_plan,
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


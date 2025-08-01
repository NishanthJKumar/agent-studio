import copy
import datetime
import logging
import readline  # noqa: F401
import sys
import time
from pathlib import Path

import numpy as np

from agent_studio.agent.base_agent import BaseAgent
from agent_studio.llm import ModelManager
from agent_studio.llm.utils import extract_from_response
from agent_studio.utils.types import Message, MessageList, StepInfo, TaskConfig

logger = logging.getLogger(__name__)


class FeedbackBasedAgent(BaseAgent):
    """Variant of the direct agent augmented with feedback."""

    name: str = "feedback"

    def __init__(
        self,
        model: str,
        remote: bool,
        runtime_server_addr: str,
        runtime_server_port: int,
        results_dir: Path,
        restrict_to_one_step: bool,
        prompt_approach: str = "naive",
        feedback_model: str = "gpt-4o-2024-08-06",
        feedback_prompt_approach: str = "direct",
        max_critique_attempts: int = 3,
        model_server: str = None,
    ) -> None:
        """Initialize everything the same way as the parent class, but also
        initialize a feedback model and buffer."""
        super().__init__(
            model=model,
            remote=remote,
            runtime_server_addr=runtime_server_addr,
            runtime_server_port=runtime_server_port,
            results_dir=results_dir,
            prompt_approach=prompt_approach,
            restrict_to_one_step=restrict_to_one_step,
        )
        with open(
            f"agent_studio/agent/prompts/{prompt_approach}_system_prompt.txt", "r"
        ) as file:
            self._system_prompt = file.read()
        self._feedback_prompt = None
        self.feedback_history: MessageList = []
        if feedback_model != "human":
            with open(
                (
                    f"agent_studio/agent/prompts/{feedback_prompt_approach}_"
                    "feedback_prompt.txt"
                ),
                "r",
            ) as file:
                self._feedback_prompt = file.read()
            feedback_model_manager = ModelManager()
            self.feedback_model = feedback_model_manager.get_model(feedback_model)
        else:
            self.feedback_model = None
        self.feedback_model_name = feedback_model
        self.feedback_prompt_approach = feedback_prompt_approach
        self._plan_criticized = False
        self.max_critique_attempts = max_critique_attempts
        if self.feedback_prompt_approach == "plan_critique":
            assert (
                prompt_approach == "bilevel_planning"
            ), "plan_critique only works with bilevel planning prompting."

    @property
    def feedback_model_prompt(self) -> MessageList:
        messages: MessageList = []
        messages.append(Message(role="system", content=self._feedback_prompt))
        messages.append(
            Message(role="user", content=f"The task instruction: {self.instruction}")
        )
        assert len(self.feedback_history) == len(self.trajectory) - 1
        for i, content_tuple in enumerate(
            zip(self.feedback_history, self.trajectory[:-1])
        ):
            past_feedback, step = content_tuple
            messages.append(
                Message(
                    role="assistant",
                    content=f"Step number: {i}.\nAction:\n\
                    ```python\n{step.action}\n```\n\n"
                    f"Error(s) from execution:\n{step.result}"
                    f"Feedback:\n{past_feedback.content}",
                )
            )
        messages.append(
            Message(
                role="assistant",
                content=f"Step number: {len(self.trajectory)}.\nAction:\
                \n```python\n{self.trajectory[-1].action}\n```\n\n"
                f"Error(s) from execution:\n{self.trajectory[-1].result}",
            )
        )
        if self.obs is not None and self.feedback_prompt_approach != "plan_critique":
            # Plan critique doesn't use observations.
            messages.append(Message(role="user", content=self.obs))
        return messages

    def reset(self, task_config: TaskConfig) -> None:
        super().reset(task_config)
        self.feedback_history = []
        self._plan_criticized = False

    def generate_action(self, obs: np.ndarray | None, model_name: str) -> StepInfo:
        """Generate an action based on the observation."""
        self.obs = obs
        prompt = self.action_prompt
        assert prompt is not None, "Invalid prompt"
        logger.info(f"Prompt: {prompt}")
        response, info = self.model.generate_response(messages=prompt, model=model_name)
        logger.info(f"Response: {response}")
        assert response is not None, "Failed to generate response."
        self.total_tokens += info.get("total_tokens", 0)
        # In the case of the plan_critique approach, we need to iterate feedback
        # until the feedback model is satisfied with the plan.
        if (
            self.feedback_prompt_approach == "plan_critique"
            and not self._plan_criticized
        ):
            for _ in range(self.max_critique_attempts):
                # Extract the action from the response and add it to the
                # trajectory - needed to setup the proper prompt for the
                # feedback model.
                action = extract_from_response(response).strip()
                step_info = StepInfo(
                    obs=obs,
                    prompt=prompt,
                    response=response,
                    action=action,
                    info=info,
                    result={},
                    timestamp=0.0,
                )
                step_info.timestamp = time.time()
                self.trajectory.append(step_info)
                # Get feedback.
                feedback_prompt = self.feedback_model_prompt
                logger.info(f"Feedback Prompt: {feedback_prompt}")
                feedback_response = self._query_feedback_model(feedback_prompt)
                logger.info(f"Feedback Response: {feedback_response}")
                self.feedback_history.append(
                    Message(role="assistant", content=feedback_response)
                )
                if len(feedback_response) > 0 and "No feedback." in feedback_response:
                    break
                else:
                    logger.info("Feedback model is not satisfied with the plan.")
                    prompt = self.action_prompt
                    assert prompt is not None
                    logger.info(f"Prompt: {prompt}")
                    response, info = self.model.generate_response(
                        messages=prompt, model=model_name
                    )
                    logger.info(f"Response: {response}")
            self._plan_criticized = True
            # We need to pop the last step from the trajectory since it
            # will get added back.
            self.trajectory.pop(-1)
        # Extract the action from the response.
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
        if self.task_config.restrict_to_one_step:
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
            info=info,
            unexecuted_code=unexecuted_code,
            result={},
            timestamp=0.0,
        )

    def step_action(
        self, failure_msg: str | None, step_info: StepInfo
    ) -> tuple[dict, bool]:
        """Same as the above method from the superclass,
        except that we also add feedback
        """
        result = {}
        done = False
        if not failure_msg:
            code_clean = step_info.action
            exit_in_code = False
            if code_clean.endswith("exit()"):
                code = code_clean[: -len("exit()")].strip()
                exit_in_code = True
            else:
                code = code_clean
            logger.debug(f"Code to execute:\n{code}\n")
            # import ipdb

            # ipdb.set_trace()
            result = self.runtime(code)
            step_info.result = copy.deepcopy(result)
            step_info.timestamp = time.time()
            self.trajectory.append(step_info)
            # Get feedback.
            if self.feedback_prompt_approach != "plan_critique":
                feedback_prompt = self.feedback_model_prompt
                logger.info(f"Prompt: {feedback_prompt}")
                response = self._query_feedback_model(feedback_prompt)
                logger.info(f"Response: {response}")
                self.feedback_history.append(
                    Message(role="assistant", content=response)
                )
                if len(result.keys()) == 0 and "No feedback." in response:
                    done = True
            else:
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

    def _query_feedback_model(self, feedback_prompt: MessageList) -> str:
        if self.feedback_model is None:
            for message in feedback_prompt[:-1]:
                print(
                    f"Message.\n role: {message.role}\n content:{message.content}\n\n"
                )
            print(
                f"Message.\n role: {feedback_prompt[-1].role}\n content:"
                f"{feedback_prompt[-1].content}"
            )
            sys.stdin.flush()
            response = input("Feedback: ")
        else:
            logger.info(f"Feedback Prompt: {feedback_prompt}")
            response, info = self.feedback_model.generate_response(
                messages=feedback_prompt, model=self.feedback_model_name
            )
            logger.info(f"Feedback Response: {response}")
        return response

    @property
    def action_prompt(self) -> MessageList:
        messages: MessageList = []
        messages.append(Message(role="system", content=self._system_prompt))
        messages.append(
            Message(
                role="user",
                content=f"The task instruction: \
            {self.instruction}",
            )
        )
        # In most cases, we want to add the feedback history to the prompt.
        if (
            self.feedback_prompt_approach != "plan_critique"
            or not self._plan_criticized
        ):
            try:
                assert len(self.feedback_history) == len(self.trajectory)
            except AssertionError:
                logger.error(
                    f"Feedback history length: {len(self.feedback_history)}\n"
                    f"Trajectory length: {len(self.trajectory)}"
                )
                raise AssertionError("Feedback history and trajectory length mismatch")
            for i, content_tuple in enumerate(
                zip(self.feedback_history, self.trajectory)
            ):
                past_feedback, step = content_tuple
                messages.append(
                    Message(
                        role="assistant",
                        content=f"##Step number: {i}.\n"
                        f"##Agent Thoughts: {step.response}\n"
                        "##Action Executed: "
                        f"```python\n{step.action}\n```\n\n"
                        f"##Unexecuted Code: {step.unexecuted_code}\n"
                        f"##Execution Output:\n{step.result}",
                    )
                )
                messages.append(
                    Message(
                        role="user",
                        content=f"Feedback:\n{past_feedback.content}",
                    )
                )
        else:
            # In this case, there is no more critic feedback to add.
            for i, step in enumerate(self.trajectory):
                messages.append(
                    Message(
                        role="assistant",
                        content=f"##Step number: {i}.\n"
                        f"##Agent Thoughts: {step.response}\n"
                        "##Action Executed: "
                        f"```python\n{step.action}\n```\n\n"
                        f"##Unexecuted Code: {step.unexecuted_code}\n"
                        f"##Execution Output:\n{step.result}",
                    )
                )
        # Finally, add in the observation.
        if self.obs is not None:
            messages.append(Message(role="user", content=self.obs))
        return messages

import copy
import logging
import time
from pathlib import Path

from agent_studio.agent.base_agent import BaseAgent
from agent_studio.llm import ModelManager
from agent_studio.utils.types import Message, MessageList, StepInfo

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a world-class programmer who can complete any instruction by executing Python code. Now you are operating a real computer-based environment, and you may be given a screenshot of the current computer screen. The only way to interact with the environment is to write Python code.
You are given a task instruction in the form of a string and you need to write Python code to complete it.
You are using Jupyter Notebook to execute the code and shell commands, so generate code/shell commands step by step with multiple blocks. You can use jupyter internal operator "%" and "!". The generated code/shell commands should be wrapped between "```python\n" and "\n```". Your response should include and only include one code block. You will get the execution result of the code.
You can interact with the Notebook multiple rounds. Thus, if you are not sure about the code, you can submit the code to see the result and modify the code if needed. When you think the instruction is completed and the code is correct, end the code block with `exit()`.
For simplicity, you can use the following code snippets:
You are probably given a screenshot of the current computer screen. You can only use the Jupyter Notebook to interact with the environment. We have provided the initial code to access the mouse and keyboard:
```python
from agent_studio.envs.desktop_env import Mouse, Keyboard
mouse = Mouse()
keyboard = Keyboard()
```
You can use the `mouse` and `keyboard` objects to interact with the environment. `mouse.click(x: int, y: int, button: str, clicks: int, interval: float)` can be used to click the "button" at the specified position "click" times with a specific interval. You can choose "button" from "left", "right", and middle". `keyboard.type(text: str, interval: float)` can be used to type the specified text with a specific interval. `keyboard.hotkey(keys: list[str])` can be used to press hotkeys.
If your task needs to access the Google service, you can use the `credentials.json` file in the `./agent_studio/config` directory. Also, there are six token files, `docs_token.json`, `drive_token.json`, `gmail_token.json`, `sheets_token.json`, `slides_token.json`, `calendar_token.json` and `forms_token.json`, in the `./agent_studio/config` directory, and you can use any of them to access the corresponding Google service.
E.g. you can use the following code to access the Google Drive API:
```python
import json
from google.oauth2 import credentials
from googleapiclient.discovery import build

token_path="agent_studio/config/docs_token.json"
with open(token_path, "r") as f:
    token = json.loads(f.read())
creds = credentials.Credentials.from_authorized_user_info(token, [
    "https://www.googleapis.com/auth/drive",
])
service = build("drive", "v3", credentials=creds)
service.files().get_media(fileId=xxxxxxx)
```
Also, you should assume the timezone is UTC+0 if there's no further specification."""  # noqa: E501


FEEDBACK_PROMPT = """You are an expert-level observer and critic watching a particular agent attempting to complete a task.
You are in the same environment as the agent, and your job is to provide descriptive, natural language critques of the agent's performance.
Importantly, do not try to solve the agent's task: just identify what the agent has attempted to do
and provide feedback on its latest action as well as the outcome.
If everything looks correct, and you expect the agent to solve the task after executing the current step, then simply output (without the quotes): 'No feedback.'
Otherwise, provide a free-form natural language description of what the problems are with the agent's current action given the history
of actions it has executed, and their results (which are provided to you).
"""  # noqa: E501


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
        feedback_model: str,
    ) -> None:
        """Initialize everything the same way as the parent class, but also
        initialize a feedback model and buffer."""
        super().__init__(
            model=model,
            remote=remote,
            runtime_server_addr=runtime_server_addr,
            runtime_server_port=runtime_server_port,
            results_dir=results_dir,
        )
        self.feedback_history: MessageList = []
        feedback_model_manager = ModelManager()
        self.feedback_model = feedback_model_manager.get_model(feedback_model)
        self.feedback_model_name = feedback_model

    @property
    def feedback_model_prompt(self) -> MessageList:
        messages: MessageList = []
        messages.append(Message(role="system", content=FEEDBACK_PROMPT))
        messages.append(
            Message(role="user", content=f"The task instruction: {self.instruction}")
        )
        assert len(self.feedback_history) == len(self.trajectory) - 1
        for past_feedback, step in zip(self.feedback_history, self.trajectory[:-1]):
            messages.append(
                Message(
                    role="assistant",
                    content=f"Action:\n```python\n{step.action}\n```\n\n"
                    f"Execution result:\n{step.result}"
                    f"Feedback:\n{past_feedback.content}",
                )
            )
        messages.append(
            Message(
                role="assistant",
                content=f"Action:\n```python\n{self.trajectory[-1].action}\n```\n\n"
                f"Execution result:\n{self.trajectory[-1].result}",
            )
        )
        return messages

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
            if code_clean.endswith("exit()"):
                code = code_clean[: -len("exit()")].strip()
            else:
                code = code_clean
            logger.debug(f"Code to execute:\n{code}\n")
            result = self.runtime(code)
            step_info.result = copy.deepcopy(result)
            step_info.timestamp = time.time()
            self.trajectory.append(step_info)
            # Get feedback.
            feedback_prompt = self.feedback_model_prompt
            response, info = self.feedback_model.generate_response(
                messages=feedback_prompt, model=self.feedback_model_name
            )
            self.feedback_history.append(Message(role="assistant", content=response))
            if len(result.keys()) == 0 and "No feedback." in response:
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
        messages.append(Message(role="system", content=SYSTEM_PROMPT))
        messages.append(
            Message(role="user", content=f"The task instruction: {self.instruction}")
        )
        assert len(self.feedback_history) == len(self.trajectory)
        for past_feedback, step in zip(self.feedback_history, self.trajectory):
            messages.append(
                Message(
                    role="assistant",
                    content=f"Action:\n```python\n{step.action}\n```\n\n"
                    f"Execution result:\n{step.result}"
                    f"Feedback:\n{past_feedback.content}",
                )
            )

        if self.obs is not None:
            messages.append(Message(role="user", content=self.obs))

        return messages

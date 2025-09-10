from pathlib import Path

from agent_studio.agent.base_agent import BaseAgent
from agent_studio.utils.types import Message, MessageList


class DirectAgent(BaseAgent):
    """Zero-shot agents."""

    name: str = "direct"

    def __init__(
        self,
        seed: int,
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
        """Initialize everything the same way as the parent class, but also
        initialize a feedback model and buffer."""
        super().__init__(
            seed=seed,
            model=model,
            remote=remote,
            runtime_server_addr=runtime_server_addr,
            runtime_server_port=runtime_server_port,
            results_dir=results_dir,
            restrict_to_one_step=restrict_to_one_step,
            prompt_approach=prompt_approach,
            model_server=model_server,
            summarization_prompt_approach=summarization_prompt_approach,
            extra_args=extra_args,
        )
        with open(
            f"agent_studio/agent/prompts/{prompt_approach}_system_prompt.txt", "r"
        ) as file:
            self._system_prompt = file.read()

    @property
    def action_prompt(self) -> MessageList:
        messages: MessageList = []
        messages.append(Message(role="system", content=self._system_prompt))
        messages.append(
            Message(role="user", content=f"The task instruction: {self.instruction}")
        )
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

        if self.obs is not None:
            messages.append(Message(role="user", content=self.obs))

        return messages

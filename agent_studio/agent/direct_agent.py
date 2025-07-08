from pathlib import Path

from agent_studio.agent.base_agent import BaseAgent
from agent_studio.utils.types import Message, MessageList


class DirectAgent(BaseAgent):
    """Zero-shot agents."""

    name: str = "direct"

    def __init__(
        self,
        model: str,
        remote: bool,
        runtime_server_addr: str,
        runtime_server_port: int,
        results_dir: Path,
        prompt_approach: str = "naive",
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

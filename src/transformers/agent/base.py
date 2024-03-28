from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ..base import Tool


class BaseAgent(ABC):

    @property
    def toolbox(self) -> List[Tool]:
        """Get all tool currently available to the agent."""

    @abstractmethod
    @classmethod
    def from_hub(
        cls,
        *args: Any,
        **kwds: Any
    ): # TODO define args
        "Creates agent with an endpoint from HF hub"
        return cls

    @abstractmethod
    def run(
        self,
        task: Dict[str, str], # TODO create Message class?
        **kwds: Any
    ) -> Any:
        "Repeatedly calls 'step()' method in a loop until the task is done."

    @abstractmethod
    def step(
        self,
        **kwds: Any
    ) -> Any:
        """Main method responsible for interacting with the LLM and updating
        the agent's state from the memory."""

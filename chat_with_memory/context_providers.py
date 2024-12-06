from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime

from atomic_agents.lib.components.system_prompt_generator import (
    SystemPromptContextProviderBase,
)

from chat_with_memory.tools.memory_models import BaseMemory


class MemoryContextProvider(SystemPromptContextProviderBase):
    """Provides context from previously stored memories relevant to the current conversation."""

    def __init__(
        self,
        title: str,
    ):
        super().__init__(title)
        self.memories: List[BaseMemory] = []

    def get_info(self) -> str:
        """
        Get the current context information.

        Returns:
            Formatted string of current memory context
        """

        output = "Timestamp | Memory Type | Content\n"
        output += "-----------------------------------\n"
        for memory in self.memories:
            output += f"{memory.timestamp} | {memory.memory_type} | {memory.content}\n"

        return output


class CurrentDateContextProvider(SystemPromptContextProviderBase):
    """Provides the current date and time."""

    def __init__(self, title: str):
        super().__init__(title)

    def get_info(self) -> str:
        return f"The current datetime in the format YYYY-MM-DD HH:MM:SS is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


if __name__ == "__main__":
    # Example usage
    memory_context_provider = MemoryContextProvider(
        title="Memory Context",
    )

    # Test with a sample query
    memory_context_provider.update_context(
        "Tell me about any previous conversations about work projects"
    )
    print(memory_context_provider.get_info())

import os
from rich.console import Console
from rich.panel import Panel
from rich.style import Style

from chat_with_memory.agents.chat_agent import (
    chat_agent,
    ChatAgentInputSchema,
    ChatAgentOutputSchema,
)
from chat_with_memory.agents.memory_formation_agent import (
    memory_formation_agent,
    MemoryFormationInputSchema,
)
from chat_with_memory.tools.memory_store_tool import (
    MemoryStoreTool,
    MemoryStoreInputSchema,
)
from chat_with_memory.tools.memory_query_tool import (
    MemoryQueryInputSchema,
    MemoryQueryTool,
)
from chat_with_memory.context_providers import (
    MemoryContextProvider,
    CurrentDateContextProvider,
)


def format_conversation_for_memory(role: str, content: str) -> str:
    """Format the conversation in a way that provides clear context for memory formation."""
    return f"{role}: {content}"


def main() -> None:
    console = Console()
    store_tool = MemoryStoreTool()

    # Define muted style for background processes
    muted_style = Style(color="grey69", dim=True)

    # Initialize tools and context providers
    memory_context_provider = MemoryContextProvider(
        title="Existing Memories",
    )
    current_date_context_provider = CurrentDateContextProvider(
        title="Current Date",
    )

    # Register context providers with agents using proper registration method
    chat_agent.register_context_provider("memory", memory_context_provider)
    chat_agent.register_context_provider("current_date", current_date_context_provider)
    memory_formation_agent.register_context_provider("memory", memory_context_provider)
    memory_formation_agent.register_context_provider(
        "current_date", current_date_context_provider
    )

    # Initial greeting
    initial_message = ChatAgentOutputSchema(response="Hello, how are you?")
    chat_agent.memory.add_message("assistant", initial_message)
    console.print(f"[bold green]Assistant:[/bold green] {initial_message.response}")

    last_assistant_msg = initial_message.response

    try:
        while True:
            # Get user input
            console.print("[bold blue]User:[/bold blue]", end=" ")
            user_input = input()

            # Use the query tool to get the memory
            memory_query_tool = MemoryQueryTool()
            retrieved_memories = memory_query_tool.run(
                MemoryQueryInputSchema(query=user_input, n_results=10)
            )

            memory_context_provider.memories = retrieved_memories.memories

            # Check if we need to form a new memory from the user input
            memory_assessment = memory_formation_agent.run(
                MemoryFormationInputSchema(
                    last_user_msg=user_input, last_assistant_msg=last_assistant_msg
                )
            )

            # Store and display any formed memories
            if memory_assessment.memories:
                console.print("\n", style=muted_style)
                console.print(
                    Panel(
                        "📝 Forming new memories...",
                        style=muted_style,
                        title="Memory Formation",
                    )
                )

                for memory in memory_assessment.memories:
                    # Store the memory
                    store_result = store_tool.run(MemoryStoreInputSchema(memory=memory))

                    # Display the stored memory in muted style
                    console.print(
                        Panel(
                            f"Type: {memory.__class__.__name__}\n"
                            f"Content: {memory.content}\n",
                            style=muted_style,
                            title="Stored Memory",
                            border_style=muted_style,
                        )
                    )
                console.print()  # Add spacing after memories

            # Process user message through chat agent
            user_message = ChatAgentInputSchema(message=user_input)
            chat_response = chat_agent.run(user_message)

            last_assistant_msg = chat_response.response

            # Display assistant's response
            console.print(
                f"[bold green]Assistant:[/bold green] {chat_response.response}"
            )

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Conversation ended. Goodbye![/bold yellow]")
    except Exception as e:
        console.print(f"\n[bold red]An error occurred: {str(e)}[/bold red]")


if __name__ == "__main__":
    main()

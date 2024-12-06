from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from typing import Any
from pathlib import Path

from chat_with_memory.tools.memory_store_tool import (
    MemoryStoreTool,
    MemoryStoreInputSchema,
)
from chat_with_memory.tools.memory_query_tool import (
    MemoryQueryTool,
    MemoryQueryInputSchema,
)
from chat_with_memory.tools.memory_models import (
    CoreBioMemory,
    EventMemory,
    WorkProjectMemory,
)


def run_demo() -> None:
    """Run the memory demo"""
    console = Console()
    store_tool = MemoryStoreTool()
    query_tool = MemoryQueryTool()

    # Example memories
    example_memories = [
        CoreBioMemory(
            content="Allergic to peanuts since childhood",
        ),
        EventMemory(
            content="First day at new job",
        ),
        WorkProjectMemory(
            content="AI Chat Project Implementation",
        ),
    ]

    # Store memories
    console.print("\n[bold blue]Storing Memories...[/bold blue]")
    stored_memories = []
    for memory in example_memories:
        result = store_tool.run(MemoryStoreInputSchema(memory=memory))
        stored_memories.append(result.memory)
        print_memory(console, result.memory, "Stored Memory")

    # Demo different query scenarios
    console.print("\n[bold blue]Demonstrating Queries...[/bold blue]")

    # Query by type
    for memory_type in ["core", "event", "work_project"]:
        console.print(
            f"\n[bold yellow]Querying {memory_type} memories...[/bold yellow]"
        )
        query_result = query_tool.run(
            MemoryQueryInputSchema(
                query="Find relevant memories", n_results=5, memory_type=memory_type
            )
        )
        for memory in query_result.memories:
            print_memory(console, memory, f"Found {memory_type} Memory")

    # Semantic search
    console.print("\n[bold yellow]Semantic Search Demo[/bold yellow]")
    query_result = query_tool.run(
        MemoryQueryInputSchema(
            query="Find work-related content",
            n_results=3,
        )
    )
    for memory in query_result.memories:
        print_memory(console, memory, "Found Memory (Semantic)")


def print_memory(
    console: Console,
    memory: CoreBioMemory | EventMemory | WorkProjectMemory,
    title: str = "Memory",
) -> None:
    """Helper function to print a memory in a nice format"""
    table = Table(show_header=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")

    # Print base properties
    table.add_row("ID", str(memory.id))
    table.add_row("Content", memory.content)
    table.add_row("Timestamp", memory.timestamp.isoformat())
    table.add_row("Type", memory.__class__.__name__)

    # Add type-specific fields if they exist
    if isinstance(memory, CoreBioMemory):
        # Assuming CoreBioMemory might have additional properties in the future
        pass
    elif isinstance(memory, EventMemory):
        # Assuming EventMemory might have additional properties in the future
        pass
    elif isinstance(memory, WorkProjectMemory):
        # Assuming WorkProjectMemory might have additional properties in the future
        pass

    console.print(Panel(table, title=title))


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()

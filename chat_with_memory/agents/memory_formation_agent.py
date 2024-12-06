import instructor
from openai import OpenAI
import os
from typing import List, Literal, Optional, Union, Dict, Any
from pydantic import Field
from datetime import datetime, timezone

from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseIOSchema
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from atomic_agents.lib.components.agent_memory import AgentMemory
from chat_with_memory.tools.memory_models import (
    BaseMemory,
    CoreBioMemory,
    EventMemory,
    WorkProjectMemory,
)
from chat_with_memory.tools.memory_store_tool import (
    MemoryStoreTool,
    MemoryStoreInputSchema,
)
from chat_with_memory.tools.memory_query_tool import (
    MemoryQueryTool,
    MemoryQueryInputSchema,
)


class MemoryFormationInputSchema(BaseIOSchema):
    """Input schema for the Memory Formation Agent."""

    last_user_msg: str = Field(
        ...,
        description="The last message from the user in the conversation",
    )
    last_assistant_msg: str = Field(
        ...,
        description="The last message from the assistant in the conversation",
    )


class MemoryFormationOutputSchema(BaseIOSchema):
    """Output schema for the Memory Formation Agent, representing the assistant's memory about the user."""

    reasoning: List[str] = Field(
        ...,
        description="Reasoning about which memory type to pick from the list of possible memory types and why",
        min_length=3,
        max_length=5,
    )
    memories: Optional[List[CoreBioMemory | EventMemory | WorkProjectMemory]] = Field(
        ...,
        description="The formed memories of the assistant about the user, if anything relevant was found.",
    )


# Initialize the system prompt generator with more selective criteria
memory_formation_prompt = SystemPromptGenerator(
    background=[
        "You are an AI specialized in identifying and preserving truly significant, long-term relevant information about users.",
        "You focus on extracting information that will remain relevant and useful over extended periods.",
        "You carefully filter out temporary states, trivial events, and time-bound information.",
        "You carefully filter out any memories that are already in the memory store.",
        "You understand the difference between temporarily relevant details and permanently useful knowledge.",
    ],
    steps=[
        "Analyze both the user's message and the assistant's message for context",
        "Consider the conversation flow to better understand the information's significance",
        "Look for information meeting these criteria:",
        "  - Permanent or long-lasting relevance (e.g., traits, background, significant relationships)",
        "  - Important biographical details (e.g., health conditions, cultural background)",
        "  - Major life events that shape the user's context",
        "  - Information that would be valuable months or years from now",
        "Filter out information that is:",
        "  - Temporary or time-bound",
        "  - Trivial daily events",
        "  - Current activities or states",
        "  - Administrative or routine matters",
        "  - Already in the existing memories",
        "For each truly significant piece of information:",
        "  - Formulate it in a way that preserves long-term relevance",
        "  - Choose the appropriate memory type",
        "  - Express it clearly and timelessly",
    ],
    output_instructions=[
        "Create memories only for information with lasting significance",
        "Do not create memories of things that are already in the memory store",
        "Format memories to be relevant regardless of when they are accessed",
        "Focus on permanent traits, important relationships, and significant events",
        "Exclude temporary states and trivial occurrences",
        "When in doubt, only store information that would be valuable in future conversations",
    ],
)

# Create the agent configuration
memory_formation_config = BaseAgentConfig(
    client=instructor.from_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY"))),
    model="gpt-4o-mini",
    memory=AgentMemory(max_messages=10),
    system_prompt_generator=memory_formation_prompt,
    input_schema=MemoryFormationInputSchema,
    output_schema=MemoryFormationOutputSchema,
)

# Create the memory formation agent
memory_formation_agent = BaseAgent(memory_formation_config)


if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    store_tool = MemoryStoreTool()
    query_tool = MemoryQueryTool()

    # Update test examples to use the new input format
    test_inputs = [
        # Core biographical memories - health
        {
            "last_assistant_msg": "Is there anything important I should know about your health?",
            "last_user_msg": "Yes, actually - I have a severe shellfish allergy that was discovered when I was 7 years old. It causes anaphylaxis and I always carry an EpiPen. I also have mild asthma that acts up during pollen season. I've been feeling a bit under the weather lately with a cold, but that's temporary.",
        },
        # Core biographical memories - background
        {
            "last_assistant_msg": "Tell me about your background.",
            "last_user_msg": "I come from a multicultural family - my mother is Japanese and my father is Brazilian. I grew up speaking three languages at home: Portuguese, Japanese, and English. I have a PhD in Quantum Computing from MIT, which I completed in 2019. I've been working in quantum cryptography ever since. My coffee machine broke this morning and I had a terrible commute, but that's just today's drama!",
        },
        # Significant events and relationships
        {
            "last_assistant_msg": "Any significant changes in your life recently?",
            "last_user_msg": "Yes, actually - I just got engaged last month to Alex, whom I met during our medical residency at Mayo Clinic. We're planning to move to Boston next year for my new position as Head of Cardiology at Mass General. My sister Maria just had her first child too - I'm an aunt now! The weather's been awful lately, and I need to get my car fixed, but those are minor concerns.",
        },
        # Work and professional development
        {
            "last_assistant_msg": "Tell me about your work.",
            "last_user_msg": "I'm leading Project Aurora, our company's quantum-resistant cryptography implementation. It's a $2.5M initiative in collaboration with IBM Quantum, targeting Q4 2024 for rollout. I manage a team of 15 people and we're pioneering new approaches in post-quantum security. My office chair is uncomfortable and the AC is acting up, but that's not important in the grand scheme of things.",
        },
    ]

    # Store formed memories
    stored_memories = []
    for input_data in test_inputs:
        memory_formation_agent.memory = AgentMemory()
        console.print(
            f"\n[bold blue]Processing Conversation:[/bold blue]\n"
            f"Assistant: {input_data['last_assistant_msg']}\n"
            f"User: {input_data['last_user_msg']}"
        )

        # Form memories using the agent with the new input schema
        response = memory_formation_agent.run(
            MemoryFormationInputSchema(
                last_assistant_msg=input_data["last_assistant_msg"],
                last_user_msg=input_data["last_user_msg"],
            )
        )

        # Display formed memories
        for memory in response.memories:
            console.print(
                Panel(
                    f"[bold green]Memory Type:[/bold green] {memory.__class__.__name__}\n"
                    f"[bold cyan]Content:[/bold cyan] {memory.content}\n"
                    f"[bold magenta]Reasoning:[/bold magenta] {', '.join(response.reasoning)}\n"
                    f"[bold yellow]Timestamp:[/bold yellow] {memory.timestamp}",
                    title="Formed Memory",
                )
            )

            # Store each memory
            store_result = store_tool.run(MemoryStoreInputSchema(memory=memory))
            stored_memories.append(store_result.memory)
        console.print()

    # Demonstrate querying memories
    console.print("\n[bold blue]Querying Stored Memories[/bold blue]")

    # Query by type examples
    memory_types = ["core", "event", "work_project"]
    for memory_type in memory_types:
        console.print(f"\n[bold yellow]Querying {memory_type} memories:[/bold yellow]")
        query_result = query_tool.run(
            MemoryQueryInputSchema(
                query="Find relevant memories", n_results=5, memory_type=memory_type
            )
        )

        for memory in query_result.memories:
            console.print(
                Panel(
                    f"[bold green]Memory Type:[/bold green] {memory.__class__.__name__}\n"
                    f"[bold cyan]Content:[/bold cyan] {memory.content}\n"
                    f"[bold yellow]Timestamp:[/bold yellow] {memory.timestamp}",
                    title=f"Retrieved {memory_type.title()} Memory",
                )
            )

    # Semantic search example
    console.print("\n[bold yellow]Semantic Search Demo:[/bold yellow]")
    search_queries = [
        "Find information about allergies",
        "Find work-related content",
        "Find meeting information",
    ]

    for query in search_queries:
        console.print(f"\n[bold cyan]Searching for:[/bold cyan] {query}")
        query_result = query_tool.run(MemoryQueryInputSchema(query=query, n_results=2))

        for memory in query_result.memories:
            console.print(
                Panel(
                    f"[bold green]Memory Type:[/bold green] {memory.__class__.__name__}\n"
                    f"[bold cyan]Content:[/bold cyan] {memory.content}\n"
                    f"[bold yellow]Timestamp:[/bold yellow] {memory.timestamp}",
                    title="Retrieved Memory (Semantic)",
                )
            )

import instructor
from openai import OpenAI
import os
from typing import List, Optional, Type
from pydantic import BaseModel, Field

from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseIOSchema
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from atomic_agents.lib.components.agent_memory import AgentMemory


class ChoiceInputSchema(BaseIOSchema):
    """Schema for input choices in the Oasis system."""

    choices: List[str] = Field(
        ..., description="List of available choices to select from"
    )
    information: str = Field(
        ..., description="Explanation or context for making the choice"
    )


class ReasoningStep(BaseModel):
    """Represents a single step in the reasoning process."""

    step_number: int = Field(..., description="The order of this reasoning step")
    description: str = Field(..., description="Description of the reasoning step")
    analysis: str = Field(..., description="Detailed analysis for this step")


class ChoiceOutputSchema(BaseIOSchema):
    """Schema for the choice output in the Oasis system."""

    reasoning_process: List[ReasoningStep] = Field(
        ..., description="Step-by-step reasoning process used to make the choice"
    )
    selected_choice: str = Field(..., description="The final selected choice")


# Initialize the system prompt generator
choice_system_prompt = SystemPromptGenerator(
    background=[
        "You are a decisive AI assistant specialized in making optimal choices.",
        "You carefully analyze all options using given context and criteria.",
        "You provide clear, logical reasoning for your decisions.",
    ],
    steps=[
        "Analyze all available choices thoroughly",
        "Consider the provided context and criteria",
        "Evaluate pros and cons of each option",
        "Make a clear decision with detailed reasoning",
    ],
    output_instructions=[
        "Break down your reasoning process into clear steps",
        "Provide specific justification for the selected choice",
        "Ensure the choice aligns with given criteria if provided",
    ],
)

# Create the agent configuration
choice_agent_config = BaseAgentConfig(
    client=instructor.from_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY"))),
    model="gpt-4o-mini",
    memory=AgentMemory(),
    system_prompt_generator=choice_system_prompt,
    input_schema=ChoiceInputSchema,
    output_schema=ChoiceOutputSchema,
)

# Create the choice agent
choice_agent = BaseAgent(choice_agent_config)


if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich.tree import Tree
    from rich.prompt import Prompt
    import time

    console = Console()

    def display_choice_results(response: ChoiceOutputSchema, step_name: str) -> None:
        """Helper function to display the results of a choice."""
        console.print(f"\n[bold blue]{step_name}[/bold blue]")

        reasoning_tree = Tree("🤔 Reasoning Process")
        for step in response.reasoning_process:
            step_node = reasoning_tree.add(
                f"[cyan]Step {step.step_number}: {step.description}[/cyan]"
            )
            step_node.add(f"[dim]{step.analysis}[/dim]")

        console.print(Panel(reasoning_tree))
        console.print("\n[bold green]Selected Choice:[/bold green]")
        console.print(Panel(response.selected_choice))
        console.print("\n[bold yellow]Explanation:[/bold yellow]")
        console.print(Panel(response.explanation))

    # Step 1: Choose the type of application to build
    console.print("\n[bold magenta]Starting Decision Tree Process[/bold magenta]")
    console.print("Step 1: Deciding on application type\n")

    app_type_response = choice_agent.run(
        ChoiceInputSchema(
            choices=[
                "Mobile App",
                "Web Application",
                "Desktop Application",
                "Command Line Tool",
            ],
            context="A startup wants to build a new productivity tool for software developers. "
            "They need to choose the most appropriate platform for their initial release.",
            criteria=[
                "Must reach the widest developer audience",
                "Should be easy to update and maintain",
                "Need to support cross-platform usage",
                "Should allow for rapid prototyping and MVP",
            ],
        )
    )

    display_choice_results(app_type_response, "Application Type Decision")
    time.sleep(1)  # Pause for readability

    # Step 2: Based on the first choice, choose the technology stack
    tech_stack_choices = {
        "Web Application": [
            "MERN (MongoDB, Express, React, Node.js)",
            "LAMP (Linux, Apache, MySQL, PHP)",
            "Python/Django with React",
            "Ruby on Rails with Vue.js",
        ],
        "Mobile App": [
            "React Native",
            "Flutter",
            "Native Android (Kotlin)",
            "Native iOS (Swift)",
        ],
        "Desktop Application": [
            "Electron",
            "Qt",
            "Python/Tkinter",
            ".NET MAUI",
        ],
        "Command Line Tool": [
            "Python with Click",
            "Rust with Clap",
            "Node.js with Commander",
            "Go with Cobra",
        ],
    }

    console.print(
        "\nStep 2: Choosing the technology stack based on previous decision\n"
    )

    tech_stack_response = choice_agent.run(
        ChoiceInputSchema(
            choices=tech_stack_choices[app_type_response.selected_choice],
            context=f"Based on the decision to build a {app_type_response.selected_choice}, "
            "we need to choose the most appropriate technology stack. "
            f"Previous decision reasoning: {app_type_response.explanation}",
            criteria=[
                "Must align with the team's existing skills",
                "Should have good community support and documentation",
                "Need to support rapid development cycles",
                "Should be scalable for future growth",
            ],
        )
    )

    display_choice_results(tech_stack_response, "Technology Stack Decision")
    time.sleep(1)  # Pause for readability

    # Step 3: Choose the deployment strategy
    console.print("\nStep 3: Determining the deployment strategy\n")

    deployment_response = choice_agent.run(
        ChoiceInputSchema(
            choices=[
                "Container-based (Docker/Kubernetes)",
                "Traditional VM deployment",
                "Serverless architecture",
                "Platform-as-a-Service (PaaS)",
            ],
            context=f"With {app_type_response.selected_choice} using "
            f"{tech_stack_response.selected_choice}, "
            "we need to determine the optimal deployment strategy. "
            f"Previous stack decision reasoning: {tech_stack_response.explanation}",
            criteria=[
                "Must be cost-effective for a startup",
                "Should support easy scaling",
                "Need to enable rapid updates and rollbacks",
                "Should provide good monitoring and logging",
            ],
        )
    )

    display_choice_results(deployment_response, "Deployment Strategy Decision")

    # Final Summary
    console.print("\n[bold magenta]Final Decision Tree Summary[/bold magenta]")
    summary_tree = Tree("🎯 Project Decisions")
    summary_tree.add(
        f"[green]Application Type:[/green] {app_type_response.selected_choice}"
    )
    summary_tree.add(
        f"[green]Technology Stack:[/green] {tech_stack_response.selected_choice}"
    )
    summary_tree.add(
        f"[green]Deployment Strategy:[/green] {deployment_response.selected_choice}"
    )

    console.print(Panel(summary_tree, title="Final Decisions"))

# Chat With Memory

A Python-based CLI chat application that demonstrates advanced conversation capabilities with persistent memory. Built using the [Atomic Agents](https://github.com/BrainBlend-AI/atomic-agents) framework, this application showcases how to create intelligent agents that can remember previous interactions and form meaningful memories about conversations.

## Features

- 🧠 **Memory Formation**: Automatically forms and stores relevant memories from conversations using a dedicated Memory Formation Agent
- 🔍 **Context-Aware Responses**: Uses ChromaDB for semantic search and retrieval of previous memories
- 📅 **Time-Aware**: Includes current date context in conversations through context providers
- 💬 **Rich CLI Interface**: Beautiful command-line interface using Rich
- 🤖 **Multi-Agent Architecture**: Leverages Atomic Agents framework for sophisticated agent interactions

## Prerequisites

- Python 3.10 or higher
- Poetry for dependency management

## Installation

1. Clone the repository
2. Install dependencies using Poetry:

```bash
poetry install
```

3. Set up your environment variables in `.env`:

```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Activate the Poetry environment:

```bash
poetry shell
```

2. Run the chat application:

```bash
python -m chat_with_memory.main
```

3. Start chatting! The assistant will:
   - Respond to your messages using the Chat Agent
   - Form memories about important aspects using the Memory Formation Agent
   - Use previous memories to provide context-aware responses
   - Make intelligent choices about responses using the Choice Agent

To exit, press `Ctrl+C`.

## Project Structure

```
chat_with_memory/
├── agents/                 # AI agents implementation
│   ├── chat_agent.py      # Handles main conversation flow
│   ├── memory_formation_agent.py  # Forms and manages memories
│   └── choice_agent.py    # Makes decisions about responses
├── tools/                 # Memory management tools
│   ├── memory_store_tool.py    # Stores memories in ChromaDB
│   ├── memory_query_tool.py    # Queries stored memories
│   └── memory_models.py        # Pydantic models for memory
├── services/              # Core services
│   └── chroma_db.py      # ChromaDB vector store implementation
├── main.py               # Application entry point
└── context_providers.py   # Provides time and memory context
```

## Technical Details

### Agents

- **Chat Agent**: Manages the main conversation flow and user interactions
- **Memory Formation Agent**: Analyzes conversations and forms relevant memories
- **Choice Agent**: Makes decisions about appropriate responses based on context

### Tools

- **Memory Store Tool**: Handles the storage of memories in ChromaDB
- **Memory Query Tool**: Performs semantic search on stored memories
- **Memory Models**: Defines Pydantic models for memory structure

### Core Technologies

- [**Atomic Agents**](https://github.com/BrainBlend-AI/atomic-agents): Framework for building and managing intelligent agents
- **OpenAI API**: Powers the language understanding and generation
- **ChromaDB**: Vector database for efficient memory storage and retrieval
- **Pydantic**: Data validation and settings management
- **Rich**: Terminal UI rendering
- **Instructor**: Enhanced OpenAI function calling

## Dependencies

Key dependencies include:
- atomic-agents ^1.0.15
- rich ^13.9.4
- instructor ^1.6.4
- openai ^1.54.4
- pydantic ^2.9.2
- chromadb ^0.5.18

## License

MIT License

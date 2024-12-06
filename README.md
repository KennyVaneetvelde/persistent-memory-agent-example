# Chat With Memory

A Python-based CLI chat application that demonstrates advanced conversation capabilities with persistent memory. The assistant can remember previous interactions and form memories about the conversation, providing more contextual and meaningful responses.

## Features

- 🧠 **Memory Formation**: Automatically forms and stores relevant memories from conversations
- 🔍 **Context-Aware Responses**: Uses ChromaDB for semantic search of previous memories
- 📅 **Time-Aware**: Includes current date context in conversations
- 💬 **Rich CLI Interface**: Beautiful command-line interface using Rich
- 🤖 **Agent-Based Architecture**: Utilizes separate agents for chat and memory formation

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
   - Respond to your messages
   - Form memories about important aspects of the conversation
   - Use previous memories to provide context-aware responses

To exit, press `Ctrl+C`.

## Project Structure

```
chat_with_memory/
├── agents/             # AI agents for chat and memory formation
├── tools/             # Tools for memory storage and querying
├── services/          # Core services
├── main.py           # Main application entry point
└── context_providers.py # Context providers for memory and date
```

## Technical Details

- Uses OpenAI's API for language processing
- ChromaDB for vector storage and retrieval of memories
- Pydantic for data validation
- Rich for terminal UI
- Atomic Agents framework for agent management

## License

MIT License

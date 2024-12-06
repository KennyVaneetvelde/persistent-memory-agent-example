from typing import Optional
from pydantic import Field
import json

from atomic_agents.lib.base.base_tool import BaseTool, BaseToolConfig
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from chat_with_memory.services.chroma_db import ChromaDBService
from chat_with_memory.tools.memory_models import (
    BaseMemory,
    CoreBioMemory,
    EventMemory,
    WorkProjectMemory,
)


class MemoryStoreInputSchema(BaseIOSchema):
    """Schema for storing memories"""

    memory: BaseMemory = Field(..., description="Memory to store")


class MemoryStoreOutputSchema(BaseIOSchema):
    """Schema for memory storage output"""

    memory: BaseMemory = Field(..., description="Stored memory with generated ID")


class MemoryStoreConfig(BaseToolConfig):
    """Configuration for the MemoryStoreTool"""

    collection_name: str = Field(
        default="chat_memories", description="Name of the ChromaDB collection to use"
    )
    persist_directory: str = Field(
        default="./chroma_db", description="Directory to persist ChromaDB data"
    )


class MemoryStoreTool(BaseTool):
    """Tool for storing chat memories using ChromaDB"""

    input_schema = MemoryStoreInputSchema
    output_schema = MemoryStoreOutputSchema

    def __init__(self, config: MemoryStoreConfig = MemoryStoreConfig()):
        super().__init__(config)
        self.db_service = ChromaDBService(
            collection_name=config.collection_name,
            persist_directory=config.persist_directory,
        )

    def run(self, params: MemoryStoreInputSchema) -> MemoryStoreOutputSchema:
        """Store a new memory in ChromaDB"""
        memory = params.memory

        # Map memory types to their storage representation
        memory_type_mapping = {
            CoreBioMemory: "core_memory",
            EventMemory: "event_memory",
            WorkProjectMemory: "work_project_memory",
        }

        # Get the specific memory type
        memory_type = memory_type_mapping.get(type(memory), "base_memory")

        # Base metadata with all values as strings
        metadata = {
            "timestamp": memory.timestamp,
            "memory_type": memory_type,
        }

        ids = self.db_service.add_documents(
            documents=[memory.content], metadatas=[metadata]
        )

        stored_memory = memory.model_copy()

        return MemoryStoreOutputSchema(memory=stored_memory)

from typing import List, Optional, Literal, Union
from pydantic import Field
from datetime import datetime
import json

from atomic_agents.lib.base.base_tool import BaseTool, BaseToolConfig
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from chat_with_memory.services.chroma_db import ChromaDBService, QueryResult
from chat_with_memory.tools.memory_models import (
    CoreBioMemory,
    EventMemory,
    WorkProjectMemory,
    BaseMemory,
)


class MemoryQueryInputSchema(BaseIOSchema):
    """Schema for querying memories"""

    query: str = Field(..., description="Query string to find relevant memories")
    n_results: Optional[int] = Field(
        default=2, description="Number of similar memories to retrieve"
    )
    memory_type: Optional[str] = Field(
        default=None, description="Optional memory type to filter memories"
    )


class MemoryQueryOutputSchema(BaseIOSchema):
    """Schema for memory query output"""

    memories: List[BaseMemory] = Field(
        default_factory=list, description="Retrieved memories"
    )


class MemoryQueryConfig(BaseToolConfig):
    """Configuration for the MemoryQueryTool"""

    collection_name: str = Field(
        default="chat_memories", description="Name of the ChromaDB collection to use"
    )
    persist_directory: str = Field(
        default="./chroma_db", description="Directory to persist ChromaDB data"
    )


class MemoryQueryTool(BaseTool):
    """Tool for querying chat memories using ChromaDB"""

    input_schema = MemoryQueryInputSchema
    output_schema = MemoryQueryOutputSchema

    def __init__(self, config: MemoryQueryConfig = MemoryQueryConfig()):
        super().__init__(config)
        self.db_service = ChromaDBService(
            collection_name=config.collection_name,
            persist_directory=config.persist_directory,
        )

    def run(self, params: MemoryQueryInputSchema) -> MemoryQueryOutputSchema:
        """Query for relevant memories using semantic search"""
        where_filter = None
        if params.memory_type:
            # Map query types to stored types
            type_mapping = {
                "core": "core_memory",
                "event": "event_memory",
                "work_project": "work_project_memory",
            }
            memory_type = type_mapping[params.memory_type]
            where_filter = {"memory_type": memory_type}

        try:
            results: QueryResult = self.db_service.query(
                query_text=params.query,
                n_results=params.n_results,
                where=where_filter,
            )

            # Map stored types back to memory classes
            memory_class_mapping = {
                "core_memory": CoreBioMemory,
                "event_memory": EventMemory,
                "work_project_memory": WorkProjectMemory,
                "base_memory": BaseMemory,
            }

            memories = []
            if results["documents"]:
                for doc, meta, id_ in zip(
                    results["documents"], results["metadatas"], results["ids"]
                ):
                    memory_type = meta.get("memory_type", "base_memory")
                    memory_class = memory_class_mapping[memory_type]

                    base_data = {
                        "id": id_,
                        "content": doc,
                        "timestamp": meta["timestamp"],
                    }
                    memories.append(memory_class(**base_data))

            return MemoryQueryOutputSchema(memories=memories)
        except Exception as e:
            print(f"Query error: {str(e)}")
            return MemoryQueryOutputSchema(memories=[])

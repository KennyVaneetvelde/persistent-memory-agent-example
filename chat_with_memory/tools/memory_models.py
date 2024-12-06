from typing import Literal, Optional
from pydantic import Field, BaseModel
from datetime import datetime, timezone
from atomic_agents.lib.base.base_io_schema import BaseIOSchema


def datetime_to_str(dt: datetime) -> str:
    """Convert datetime to ISO format string."""
    return dt.isoformat()


class BaseMemory(BaseIOSchema):
    """Base class for all memory types"""

    content: str = Field(..., description="Content of the memory")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO format timestamp of when the memory was created",
    )


class CoreBioMemory(BaseMemory):
    """This memory contains core biographical information about the user, such as their name, age, and occupation"""

    memory_type: Literal["core_bio"] = Field(default="core_bio")


class EventMemory(BaseMemory):
    """This memory contains information about an event that the user has experienced, such as a meeting, a conversation, or a significant experience"""

    memory_type: Literal["event"] = Field(default="event")


class WorkProjectMemory(BaseMemory):
    """This memory contains information about a work or project that the user has been involved in, such as a project, a task, or a significant experience"""

    memory_type: Literal["work_project"] = Field(default="work_project")

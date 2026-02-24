from typing import List, Optional

from pydantic import BaseModel, Field


class NodeInfo(BaseModel):
    node_id: str
    parent_id: Optional[str] = None
    children_ids: List[str] = Field(default_factory=list)
    level: int = 0

    def serialize(self) -> str:
        return self.model_dump_json()

    @classmethod
    def deserialize(cls, data: str) -> "NodeInfo":
        return cls.model_validate_json(data)

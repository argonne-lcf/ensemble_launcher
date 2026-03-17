from typing import List, Optional, Dict

from pydantic import BaseModel, Field


class NodeInfo(BaseModel):
    node_id: str
    secret_id: str
    parent_id: Optional[str] = None
    parent_secret_id: Optional[str] = None
    children_ids: List[str] = Field(default_factory=list)
    children_secret_ids: Dict[str, str] = Field(default_factory=dict)
    level: int = 0

    def serialize(self) -> str:
        return self.model_dump_json()

    @classmethod
    def deserialize(cls, data: str) -> "NodeInfo":
        return cls.model_validate_json(data)

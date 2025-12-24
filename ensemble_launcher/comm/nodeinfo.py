from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class NodeInfo:
    node_id:str
    parent_id: Optional[str] =  None
    children_ids: List[str] =  field(default_factory=list)
    level: int = 0

from typing import Union, List, Dict, Optional
import logging
from dataclasses import dataclass, field
from ensemble_launcher.comm import NodeInfo


logger = logging.getLogger(__name__)
    
class Node:
    """
    A simple tree representation
    """
    def __init__(self, 
                 node_id:str,
                 parent: Optional[NodeInfo] = None,
                 children:Optional[Dict[str, NodeInfo]] = None):
        self.node_id = node_id
        self.parent: NodeInfo = parent
        self.children: Dict[str,NodeInfo] = children if children is not None else {}
        
        self._level = 0 if self.parent is None else self.parent.level + 1

    @property
    def level(self):
        return self._level
    
    def set_parent(self, parent: NodeInfo):
        self.parent = parent
        self._level = parent.level + 1

    def add_child(self, child_id: str, child: NodeInfo):
        if child_id not in self.children:
            self.children[child_id] = child
        else:
            logger.debug(f"Child {child_id} already exists")

    def remove_child(self, child_id: str):
        if child_id in self.children:
            del self.children[child_id]
        else:
            logger.error(f"Child {child_id} does not exist")
            raise
    
    def info(self):
        return NodeInfo(
            node_id=self.node_id,
            parent_id=self.parent.node_id if self.parent is not None else None,
            children_ids=list(self.children.keys()) if self.children else [],
            level = self.level
        )
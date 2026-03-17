from typing import Union, List, Dict, Optional
import uuid
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
        self.children: Dict[str, NodeInfo] = children if children is not None else {}
        
        self._level = 0 if self.parent is None else self.parent.level + 1

        self._secret_id = str(uuid.uuid4())

    @property
    def level(self):
        return self._level
    
    def set_parent(self, parent: NodeInfo):
        self.parent = parent
        self._level = parent.level + 1

    def add_child(self, child_id: str, child: NodeInfo):
        if child_id in self.children:
            logger.warning(f"Child {child_id} already exists. Replacing it with new node info")
        self.children[child_id] = child

    def remove_child(self, child_id: str):
        if child_id in self.children:
            del self.children[child_id]
        else:
            logger.error(f"Child {child_id} does not exist")
            raise
    
    def info(self):
        return NodeInfo(
            node_id=self.node_id,
            secret_id=self._secret_id,
            parent_id=self.parent.node_id if self.parent is not None else None,
            parent_secret_id=self.parent.secret_id if self.parent is not None else None,
            children_ids=list(self.children.keys()) if self.children else [],
            children_secret_ids={child_id: child.secret_id for child_id,child in self.children.items()}, 
            level = self.level
        )
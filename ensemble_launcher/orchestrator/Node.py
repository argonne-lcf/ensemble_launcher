from typing import Union, List, Dict
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
                 parent: "Node" = None,
                 children:Dict = {}):
        self.node_id = node_id
        self.parent: "Node" = parent
        self.children: Dict[str,"Node"] = children
        self._comm = None

    @property
    def comm(self):
        return self._comm
    
    def add_parent(self, parent_id: str, parent: "Node"):
        if parent_id not in self.parents:
            self.parents[parent_id] = parent
        else:
            logger.debug(f"Parent {parent_id} already exists")

    def remove_parent(self, parent_id: str):
        if parent_id in self.parents:
            del self.parents[parent_id]
        else:
            logger.error(f"Parent {parent_id} does not exist")
            raise

    def add_child(self, child_id: str, child: "Node"):
        if child_id not in self.children:
            self.children[child_id] = child
        else:
            logger.debug(f"Child {child_id} already exists")

    def remove_child(self, child_id: Union[int, str]):
        if child_id in self.children:
            del self.children[child_id]
        else:
            logger.error(f"Child {child_id} does not exist")
            raise
    
    def info(self):
        return NodeInfo(
            node_id=self.node_id,
            parent_id=self.parent.node_id if self.parent is not None else None,
            children_ids=list(self.children.keys())
        )
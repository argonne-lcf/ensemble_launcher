import multiprocessing
import logging
import abc
import time

"""
This class is written to abstract away the communications between workers and childs
"""
class Node(abc.ABC):
    def __init__(self, 
                 node_id:str, 
                 my_tasks:dict,
                 my_nodes:list,
                 sys_info:dict,
                 comm_config:dict, 
                 logger=True,
                 logging_level=logging.INFO,
                 update_interval:int=None):
        self.node_id = node_id
        self.my_tasks = my_tasks
        self.my_nodes = my_nodes
        self.sys_info = sys_info
        self.logging_level = logging_level
        self.update_interval = update_interval
        self.last_update_time =time.time()

        self.comm_config = comm_config
        assert comm_config["comm_layer"] in ["multiprocessing","dragon"]
        self.parents = {}
        self.children = {}
        if logger:
            self.configure_logger()
        else:
            self.logger = None
    
    def configure_logger(self,logging_level=logging.INFO):
        self.logger = logging.getLogger(f"Node-{self.node_id}")
        handler = logging.FileHandler(f'./outputs/Node-{self.node_id}.txt', mode='w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging_level)

    def send_to_parents(self, data) -> int:
        for parent_id, pipe in self.parents.items():
            pipe.send(data)
            if self.logger:
                self.logger.debug(f"Sent message to parent {parent_id}")
        return 0

    def recv_from_parents(self, timeout: int = 60) -> list:
        messages = []
        for parent_id, pipe in self.parents.items():
            if pipe.poll(timeout):
                msg = pipe.recv()
                messages.append(msg)
                if self.logger:
                    self.logger.debug(f"Received message from parent {parent_id}")
        return messages

    def send_to_children(self, data) -> int:
        for child_id, pipe in self.children.items():
            pipe.send(data)
            if self.logger:
                self.logger.debug(f"Sent message to child {child_id}")
        return 0

    def recv_from_children(self, timeout: int = 60) -> list:
        messages = []
        for child_id, pipe in self.children.items():
            if pipe.poll(timeout):
                msg = pipe.recv()
                messages.append(msg)
                if self.logger:
                    self.logger.debug(f"Received message from child {child_id}")
        return messages

    def send_to_parent(self, parent_id: int, data) -> int:
        if parent_id in self.parents:
            self.parents[parent_id].send(data)
            if self.logger:
                self.logger.debug(f"Sent message to parent {parent_id}")
            return 0
        else:
            return 1

    def recv_from_parent(self, parent_id: int, timeout: int = 60):
        if parent_id in self.parents and self.parents[parent_id].poll(timeout):
            msg = self.parents[parent_id].recv()
            if self.logger:
                self.logger.debug(f"Received message from parent {parent_id}")
            return msg
        else:
            return None

    def send_to_child(self, child_id: int, message: str) -> int:
        if child_id in self.children:
            self.children[child_id].send(message)
            if self.logger:
                self.logger.debug(f"Sent message to child {child_id}")
            return 0
        else:
            return 1
        
    def recv_from_child(self, child_id: int, timeout: int = 60) -> str:
        if child_id in self.children and self.children[child_id].poll(timeout):
            msg = self.children[child_id].recv()
            if self.logger:
                self.logger.debug(f"Received message from child {child_id}")
            return msg
        else:
            return None
    
    def add_parent(self, parent_id: int, pipe):
        if parent_id not in self.parents:
            self.parents[parent_id] = pipe
            if self.logger:
                self.logger.debug(f"Added parent {parent_id}")
        else:
            if self.logger:
                self.logger.debug(f"Parent {parent_id} already exists")

    def remove_parent(self, parent_id: int):
        if parent_id in self.parents:
            del self.parents[parent_id]
            if self.logger:
                self.logger.debug(f"Removed parent {parent_id}")
        else:
            if self.logger:
                self.logger.debug(f"Parent {parent_id} does not exist")

    def add_child(self, child_id: int, pipe):
        if child_id not in self.children:
            self.children[child_id] = pipe
            if self.logger:
                self.logger.debug(f"Added child {child_id}")
        else:
            if self.logger:
                self.logger.debug(f"Child {child_id} already exists")

    def remove_child(self, child_id: int):
        if child_id in self.children:
            del self.children[child_id]
            if self.logger:
                self.logger.debug(f"Removed child {child_id}")
        else:
            if self.logger:
                self.logger.debug(f"Child {child_id} does not exist")

    def close(self):
        for parent_id, pipe in self.parents.items():
            pipe.close()
            if self.logger:
                self.logger.debug(f"Closed parent {parent_id}")
        for child_id, pipe in self.children.items():
            pipe.close()
            if self.logger:
                self.logger.debug(f"Closed child {child_id}")
    
    def blocking_recv_from_parent(self, parent_id: int):
        """
        Blocking receive from a specific parent. Waits indefinitely until a message is available.
        """
        if parent_id in self.parents:
            msg = self.parents[parent_id].recv()  # This blocks until a message is available
            if self.logger:
                self.logger.debug(f"Received message from parent {parent_id} (blocking)")
            return msg
        else:
            if self.logger:
                self.logger.debug(f"Cannot receive: Parent {parent_id} does not exist")
            return None
    
    def blocking_recv_from_child(self, child_id: int):
        """
        Blocking receive from a specific child. Waits indefinitely until a message is available.
        """
        if child_id in self.children:
            msg = self.children[child_id].recv()  # This blocks until a message is available
            if self.logger:
                self.logger.debug(f"Received message from parent {child_id} (blocking)")
            return msg
        else:
            if self.logger:
                self.logger.debug(f"Cannot receive: Parent {child_id} does not exist")
            return None
        
    def flush_child_pipe(self, child_id: int) -> int:
        """
        Flush a child pipe by reading and discarding all available messages.
        """
        if child_id not in self.children:
            if self.logger:
                self.logger.debug(f"Cannot flush: Child {child_id} does not exist")
            return -1
        
        count = 0
        pipe = self.children[child_id]
        
        # Read all available messages without blocking
        while pipe.poll(0):  # timeout of 0 means non-blocking
            _ = pipe.recv()  # discard the received message
            count += 1
        
        if self.logger:
            self.logger.debug(f"Flushed {count} messages from child {child_id}")
        
        return count


    @abc.abstractmethod
    def delete_tasks(self):
        """
        Abstract method that must be implemented by subclasses.
        These should only modify the dictionaries
        """
        pass

    @abc.abstractmethod
    def add_tasks(self):
        """
        Abstract method that adds tasks to children.
        This should only modify the dictonaries
        """
        pass

    @abc.abstractmethod
    def commit_task_update(self):
        """
        abstract method that can send update signals to children to update tasks
        """
        pass
from __future__ import annotations
import multiprocessing as mp
import logging
import abc
import time
import socket
try:
    import zmq
    ZMQ_AVAILABLE = True
except:
    ZMQ_AVAILABLE = False

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
        assert comm_config["comm_layer"] in ["multiprocessing","dragon","zmq"]
        self.parents = {} ##dict of node objects
        self.children = {} ##dict of node objects
        if logger:
            self.configure_logger()
        else:
            self.logger = None
        
        if self.comm_config["comm_layer"] in ["multiprocessing","dragon"]:
            ##always create pipes for multiprocessing or dragon
            my_conn, other_conn = mp.Pipe(duplex=True)
            ##add this to comm_config
            self.comm_config["my_conn"] = my_conn
            self.comm_config["other_conn"] = other_conn
        elif self.comm_config["comm_layer"] == "zmq":
            assert ZMQ_AVAILABLE, "zmq not available"
            assert "role" in self.comm_config and self.comm_config["role"] in ["parent","child"]
            self.zmq_context = zmq.Context()
            if self.comm_config["role"] == "parent":
                self.zmq_socket = self.zmq_context.socket(zmq.ROUTER)
                parent_address = self.comm_config.get("parent-address",f"{socket.gethostname()}:5555")
                self.zmq_socket.bind(f"tcp://{parent_address}")
                self.comm_config["parent-address"] = parent_address
            else:
                assert "parent-address" in self.comm_config, "Child needs parent-address"
                self.zmq_socket = self.zmq_context.socket(zmq.DEALER)
                self.zmq_socket.setsockopt(zmq.IDENTITY,f"{node_id}".encode())
                if self.logger: self.logger.info(f"connecting to:{self.comm_config['parent-address']}")
                self.zmq_socket.connect(f"tcp://{self.comm_config['parent-address']}")
                self.zmq_socket.send_multipart([f"{node_id}".encode(),b"READY"])
        else:
            self.zmq_context = None
            self.zmq_socket = None

    def configure_logger(self,logging_level=logging.INFO):
        self.logger = logging.getLogger(f"Node-{self.node_id}")
        handler = logging.FileHandler(f'./outputs/Node-{self.node_id}.txt', mode='w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging_level)

    def send_to_parent(self, parent_id: int, data) -> int:
        assert parent_id == 0
        if parent_id in self.parents:
            if self.comm_config["comm_layer"] in ["multiprocessing", "dragon"]:
                self.comm_config["other_conn"].send(data)
            elif self.comm_config["comm_layer"] == "zmq":
                self.zmq_socket.send_multipart([f"{parent_id}".encode(), data])
            if self.logger:
                self.logger.debug(f"Sent message to parent {parent_id}")
            return 0
        else:
            return 1

    def recv_from_parent(self, parent_id: int, timeout: int = 60):
        assert parent_id == 0
        if parent_id in self.parents:
            if self.comm_config["comm_layer"] in ["multiprocessing", "dragon"]:
                if self.comm_config["other_conn"].poll(timeout):
                    msg = self.comm_config["other_conn"].recv()
                    if self.logger:
                        self.logger.debug(f"Received message from parent {parent_id}")
                    return msg
            elif self.comm_config["comm_layer"] == "zmq":
                try:
                    msg = self.zmq_socket.recv_multipart(zmq.NOBLOCK)
                    if self.logger:
                        self.logger.debug(f"Received message from parent {parent_id}")
                    return msg
                except zmq.ZMQError:
                    pass
        return None

    def send_to_child(self, child_id: int, message) -> int:
        if child_id in self.children:
            if self.comm_config["comm_layer"] in ["multiprocessing", "dragon"]:
                self.children[child_id].comm_config["my_conn"].send(message)
            elif self.comm_config["comm_layer"] == "zmq":
                self.zmq_socket.send_multipart([f"{child_id}".encode(), message])
            if self.logger:
                self.logger.debug(f"Sent message to child {child_id}")
            return 0
        else:
            return 1
        
    def recv_from_child(self, child_id: int, timeout: int = 60):
        if child_id in self.children:
            if self.comm_config["comm_layer"] in ["multiprocessing", "dragon"]:
                if self.children[child_id].comm_config["my_conn"].poll(timeout):
                    msg = self.children[child_id].comm_config["my_conn"].recv()
                    if self.logger:
                        self.logger.debug(f"Received message from child {child_id}")
                    return msg
            elif self.comm_config["comm_layer"] == "zmq":
                try:
                    msg = self.zmq_socket.recv_multipart(zmq.NOBLOCK)
                    if self.logger:
                        self.logger.debug(f"Received message from child {child_id}")
                    return msg
                except zmq.ZMQError:
                    pass
        return None
    
    def blocking_recv_from_parent(self, parent_id: int):
        """
        Blocking receive from a specific parent. Waits indefinitely until a message is available.
        """
        assert parent_id == 0
        if parent_id in self.parents:
            if self.comm_config["comm_layer"] in ["multiprocessing", "dragon"]:
                msg = self.comm_config["other_conn"].recv()  # Blocking call
                if self.logger:
                    self.logger.debug(f"Received message from parent {parent_id} (blocking)")
                return msg
            elif self.comm_config["comm_layer"] == "zmq":
                msg = self.zmq_socket.recv_multipart()  # Blocking call
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
            if self.comm_config["comm_layer"] in ["multiprocessing", "dragon"]:
                msg = self.children[child_id].comm_config["my_conn"].recv()  # Blocking call
                if self.logger:
                    self.logger.debug(f"Received message from child {child_id} (blocking)")
                return msg
            elif self.comm_config["comm_layer"] == "zmq":
                msg = self.zmq_socket.recv_multipart()  # Blocking call
                if self.logger:
                    self.logger.debug(f"Received message from child {child_id} (blocking)")
                return msg
        else:
            if self.logger:
                self.logger.debug(f"Cannot receive: Child {child_id} does not exist")
            return None
        
    def send_to_parents(self, data) -> int:
        for parent_id, pipe in self.parents.items():
            self.send_to_parent(parent_id,data)
            if self.logger:
                self.logger.debug(f"Sent message to parent {parent_id}")
        return 0

    def recv_from_parents(self, timeout: int = 60) -> list:
        messages = []
        for parent_id in self.parents.keys():
            msg = self.recv_from_parent(parent_id, timeout)
            if msg is not None:
                messages.append(msg)
        return messages

    def send_to_children(self, data) -> int:
        for child_id in self.children.keys():
            self.send_to_child(child_id, data)
        return 0

    def recv_from_children(self, timeout: int = 60) -> list:
        messages = []
        for child_id in self.children.keys():
            msg = self.recv_from_child(child_id, timeout)
            if msg is not None:
                messages.append(msg)
        return messages
    
    def add_parent(self, parent_id: int, parent: Node):
        if parent_id not in self.parents:
            self.parents[parent_id] = parent
            if self.logger:
                self.logger.debug(f"Added parent {parent_id}")
        else:
            if self.logger: self.logger.warning(f"Parent {parent_id} already exists")

    def remove_parent(self, parent_id: int):
        if parent_id in self.parents:
            del self.parents[parent_id]
            if self.logger: self.logger.debug(f"Removed parent {parent_id}")
        else:
            if self.logger: self.logger.debug(f"Parent {parent_id} does not exist")

    def add_child(self, child_id: int, child: Node):
        if child_id not in self.children:
            self.children[child_id] = child
            if self.logger: self.logger.debug(f"Added child {child_id}")
        else:
            if self.logger: self.logger.debug(f"Child {child_id} already exists")

    def remove_child(self, child_id: int):
        if child_id in self.children:
            del self.children[child_id]
            if self.logger: self.logger.debug(f"Removed child {child_id}")
        else:
            if self.logger: self.logger.debug(f"Child {child_id} does not exist")

    def close(self):
        for parent_id, pipe in self.parents.items():
            pipe.close()
            if self.logger:
                self.logger.debug(f"Closed parent {parent_id}")
        for child_id, pipe in self.children.items():
            pipe.close()
            if self.logger:
                self.logger.debug(f"Closed child {child_id}")
        
    # def flush_child_pipe(self, child_id: int) -> int:
    #     """
    #     Flush a child pipe by reading and discarding all available messages.
    #     """
    #     if child_id not in self.children:
    #         if self.logger:
    #             self.logger.debug(f"Cannot flush: Child {child_id} does not exist")
    #         return -1
        
    #     count = 0
    #     pipe = self.children[child_id]
        
    #     # Read all available messages without blocking
    #     while pipe.poll(0):  # timeout of 0 means non-blocking
    #         _ = pipe.recv()  # discard the received message
    #         count += 1
        
    #     if self.logger:
    #         self.logger.debug(f"Flushed {count} messages from child {child_id}")
        
    #     return count


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
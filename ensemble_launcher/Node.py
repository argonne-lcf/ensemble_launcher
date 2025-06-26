from __future__ import annotations
from typing import Union, List, Dict
import multiprocessing as mp
import logging
import abc
import time
import socket
import random
import pickle
import sys
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
                 my_tasks:dict={},
                 my_nodes:list=[],
                 sys_info:dict={},
                 comm_config:dict={"comm_layer":"multiprocessing"},
                 logger=True,
                 logging_level=logging.INFO,
                 update_interval:int=None,
                 heartbeat_interval:int=1):
        self.node_id = node_id
        self.my_tasks = my_tasks
        self.my_nodes = my_nodes
        self.sys_info = sys_info
        self.logging_level = logging_level
        self.update_interval = update_interval
        self.last_update_time = time.time()
        self.last_heartbeat_time = None
        self.heartbeat_interval = heartbeat_interval

        self.comm_config = comm_config
        assert comm_config["comm_layer"] in ["multiprocessing","dragon","zmq"]
        self.parents = {} ##dict of node objects
        self.children = {} ##dict of node objects
        if logger:
            self.configure_logger()
        else:
            self.logger = None
        
        if self.comm_config["comm_layer"] in ["multiprocessing","dragon"]:
            ##add this to comm_config
            self._other_conn,self._my_conn = mp.Pipe(duplex=True)
        elif self.comm_config["comm_layer"] == "zmq":
            assert ZMQ_AVAILABLE, "zmq not available"
            assert "role" in self.comm_config and self.comm_config["role"] in ["parent","child"]
            self.zmq_context = None
            self.router_socket = None
            self.dealer_socket = None
            self.parent_address = None
            self.my_address = None
            self.router_cache = None
            self.router_poller = None
            self.dealer_poller = None
        else:
            self._my_conn = None
            self._other_conn = None
            self.zmq_context = None
            self.zmq_socket = None

    def setup_zmq_sockets(self):
        self.zmq_context = zmq.Context()
        if self.comm_config["role"] == "parent":
            self.router_socket = self.zmq_context.socket(zmq.ROUTER)
            self.my_address = self.comm_config.get("parent-address",f"{socket.gethostname() if 'local' not in socket.gethostname() else 'localhost'}:5555")
            self.router_socket.setsockopt(zmq.IDENTITY,f"{self.node_id}".encode())
            self.router_cache = {}
            try:
                self.router_socket.bind(f"tcp://{self.my_address}")
                if self.logger: self.logger.info(f"Successfully bound to {self.my_address}")
            except zmq.error.ZMQError as e:
                if "Address already in use" in str(e):
                    # Try binding up to 3 times with different ports
                    max_attempts = 3
                    for attempt in range(max_attempts):
                        try:
                            port = int(self.my_address.split(':')[-1]) + random.randint(1, 1000)
                            if self.logger: self.logger.info(f"Attempt {attempt+1}/{max_attempts}: Trying to bind to port {port} instead.")
                            self.my_address = f"{self.my_address.rsplit(':', 1)[0]}:{port}"
                            self.router_socket.bind(f"tcp://{self.my_address}")
                            if self.logger: self.logger.info(f"Successfully bound to {self.my_address}")
                            break  # Break out of the retry loop if binding succeeds
                        except zmq.error.ZMQError as retry_error:
                            if "Address already in use" in str(retry_error) and attempt < max_attempts - 1:
                                if self.logger: self.logger.warning(f"Port {port} also in use, retrying...")
                                continue
                            else:
                                raise retry_error
                else:
                    raise e
            self.router_poller = zmq.Poller()
            self.router_poller.register(self.router_socket, zmq.POLLIN)
        if self.parent_address is not None:
            self.dealer_socket = self.zmq_context.socket(zmq.DEALER)
            self.dealer_socket.setsockopt(zmq.IDENTITY,f"{self.node_id}".encode())
            if self.logger: self.logger.info(f"connecting to:{self.parent_address}")
            self.dealer_socket.connect(f"tcp://{self.parent_address}")
            self.dealer_poller = zmq.Poller()
            self.dealer_poller.register(self.dealer_socket, zmq.POLLIN)
            self.dealer_socket.send(pickle.dumps("READY"))
            msg = pickle.loads(self.dealer_socket.recv())
            if msg == "CONTINUE":
                if self.logger: self.logger.info(f"Received continue from parent")
            elif msg == "STOP":
                if self.logger: self.logger.info(f"Received stop from parent, Quitting...")
                sys.exit(0)
            else:
                if isinstance(msg, dict):
                    self.my_tasks.update(msg)
                else:
                    if self.logger: 
                        self.logger.warning(f"Unexpected message from parent: {msg}. Expected dict or 'CONTINUE'/'STOP'.")
                        
                    

    def configure_logger(self,logging_level=logging.INFO):
        self.logger = logging.getLogger(f"Node-{self.node_id}")
        handler = logging.FileHandler(f'./outputs/Node-{self.node_id}.txt', mode='w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging_level)

    def send_to_parent(self, parent_id: Union[int, str], data) -> int:
        if self.comm_config["comm_layer"] in ["multiprocessing", "dragon"]:
            if self.logger:
                my_fd = self._my_conn.fileno() if self._my_conn else 'None'
                other_fd = self._other_conn.fileno() if self._other_conn else 'None'
                self.logger.debug(f"send_to_parent: node {self.node_id} using pipes - my_conn={id(self._my_conn)} (fd={my_fd}), other_conn={id(self._other_conn)} (fd={other_fd})")
            self._my_conn.send(data)
        elif self.comm_config["comm_layer"] == "zmq":
            if self.dealer_socket is not None:
                self.dealer_socket.send(pickle.dumps(data))
            else:
                if self.logger:
                    self.logger.warning(f"Cannot send to parent {parent_id}: dealer_socket is not initialized.")
                return 1
        if self.logger:
            self.logger.debug(f"Sent message to parent {parent_id}: {data}")
        return 0

    def recv_from_parent(self, parent_id: Union[int, str], timeout: int = 60):
        if self.comm_config["comm_layer"] in ["multiprocessing", "dragon"]:
            if self._my_conn.poll(timeout):
                msg = self._my_conn.recv()
                if self.logger:
                    self.logger.debug(f"Received message {msg} from parent {parent_id}.")
                return msg
        elif self.comm_config["comm_layer"] == "zmq":
            if self.dealer_socket is not None:
                socks = dict(self.dealer_poller.poll(timeout * 1000))  # convert timeout to milliseconds
                if self.dealer_socket in socks and socks[self.dealer_socket] == zmq.POLLIN:
                    msg = pickle.loads(self.dealer_socket.recv())
                    if self.logger:
                        self.logger.debug(f"Received message {msg} from parent {parent_id}.")
                    return msg
            else:
                if self.logger:
                    self.logger.warning(f"Cannot receive from parent {parent_id}: dealer_socket is not initialized.")
                raise RuntimeError("dealer_socket is not initialized")
        if self.logger:
            self.logger.debug(f"No message received from parent {parent_id} within timeout {timeout} seconds.")
        return None

    def send_to_child(self, child_id: Union[int, str], message) -> int:
        if child_id in self.children:
            if self.comm_config["comm_layer"] in ["multiprocessing", "dragon"]:
                self.children[child_id]._other_conn.send(message)
            elif self.comm_config["comm_layer"] == "zmq":
                self.router_socket.send_multipart([f"{child_id}".encode(), pickle.dumps(message)])
            if self.logger:
                self.logger.debug(f"Sent message to child {child_id}")
            return 0
        else:
            return 1

    def recv_from_child(self, child_id: Union[int, str], timeout: int = 60):
        if child_id in self.children:
            if self.comm_config["comm_layer"] in ["multiprocessing", "dragon"]:
                if self.logger:
                    my_fd = self.children[child_id]._my_conn.fileno() if self.children[child_id]._my_conn else 'None'
                    other_fd = self.children[child_id]._other_conn.fileno() if self.children[child_id]._other_conn else 'None'
                    self.logger.debug(f"recv_from_child pipe {child_id}: other_conn={id(self.children[child_id]._other_conn)} (fd={other_fd}), my_conn={id(self.children[child_id]._my_conn)} (fd={my_fd})")
                if self.children[child_id]._other_conn.poll(timeout):
                    msg = self.children[child_id]._other_conn.recv()
                    if self.logger:
                        self.logger.debug(f"Received message {msg} from child {child_id}.")
                    return msg
            elif self.comm_config["comm_layer"] == "zmq":
                if child_id in self.router_cache and len(self.router_cache[child_id]) > 0:
                    msg = self.router_cache[child_id].pop(0)  # Get the first cached message
                    if self.logger:
                        self.logger.debug(f"Received cached message from child {child_id}. {msg}")
                    return msg
                tstart = time.time()
                while time.time() - tstart < timeout:
                    socks = dict(self.router_poller.poll(100)) #wait for 100ms
                    if self.router_socket in socks and socks[self.router_socket] == zmq.POLLIN:
                        msg = self.router_socket.recv_multipart()
                        msg[0] = msg[0].decode()  # Convert bytes to string for child_id
                        if self.logger: self.logger.debug(f"Received message from child {msg[0]} (expected {child_id})")
                        msg[1] = pickle.loads(msg[1])  # Unpickle the message
                        # wait for a message from the child
                        if msg[0] == str(child_id):
                            if self.logger:
                                self.logger.debug(f"Received message {msg[1]} from child {child_id}.")
                            return msg[1]
                        else:
                            if self.logger:
                                self.logger.debug(f"Received message from child {msg[0]}, but expected {child_id}. Caching the message.")
                            if msg[0] not in self.router_cache:
                                self.router_cache[msg[0]] = []
                            self.router_cache[msg[0]].append(msg[1])
        else:
            if self.logger:
                self.logger.debug(f"Cannot receive from child {child_id}: child does not exist.")
            raise ValueError(f"Child {child_id} does not exist.")
        if self.logger:
            self.logger.debug(f"No message received from child {child_id} within timeout {timeout} seconds.")
        return None

    def blocking_recv_from_parent(self, parent_id: Union[int, str]):
        """
        Blocking receive from a specific parent. Waits indefinitely until a message is available.
        """
        if self.logger: self.logger.debug(f"Waiting for message from parent {parent_id}......")
        if self.comm_config["comm_layer"] in ["multiprocessing", "dragon"]:
            msg = self._my_conn.recv()  # Blocking call
        elif self.comm_config["comm_layer"] == "zmq":
            msg = pickle.loads(self.dealer_socket.recv())  # Blocking call
        if self.logger: self.logger.debug(f"Received message from parent {parent_id} (blocking)")
        return msg

    def blocking_recv_from_child(self, child_id: Union[int, str]):
        """
        Blocking receive from a specific child. Waits indefinitely until a message is available.
        """
        if child_id in self.children:
            if self.logger: self.logger.debug(f"Waiting for message from child {child_id}......")
            if self.comm_config["comm_layer"] in ["multiprocessing", "dragon"]:
                msg = self.children[child_id]._other_conn.recv()
            elif self.comm_config["comm_layer"] == "zmq":
                if child_id in self.router_cache and len(self.router_cache[child_id]) > 0:
                    if self.logger: self.logger.debug(f"Child {child_id} has cached messages.")
                    msg = self.router_cache[child_id].pop(0)  # Get the first cached message
                while True:
                    msgs = self.zmq_socket.recv_multipart()  # Blocking call
                    child_id_in = msgs[0].decode()  # Convert bytes to string for child_id
                    msg = pickle.loads(msgs[1])  # Unpickle the message
                    if child_id_in == str(child_id):
                        break
                    else:
                        if child_id_in not in self.router_cache:
                            self.router_cache[child_id_in] = []
                        self.router_cache[child_id_in].append(msg)
                    time.sleep(0.1)  # Avoid busy waiting
        else:
            if self.logger:
                self.logger.debug(f"Cannot receive from child {child_id}: child does not exist.")
            raise ValueError(f"Child {child_id} does not exist.")
        if self.logger: self.logger.debug(f"Received message {msg} from child {child_id}")
        return msg

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

    def add_parent(self, parent_id: Union[int, str], parent: Node):
        if parent_id not in self.parents:
            self.parents[parent_id] = parent
            if self.logger:
                self.logger.debug(f"Added parent {parent_id}")
        else:
            if self.logger: self.logger.warning(f"Parent {parent_id} already exists")

    def remove_parent(self, parent_id: Union[int, str]):
        if parent_id in self.parents:
            del self.parents[parent_id]
            if self.logger: self.logger.debug(f"Removed parent {parent_id}")
        else:
            if self.logger: self.logger.debug(f"Parent {parent_id} does not exist")

    def add_child(self, child_id: Union[int, str], child: Node):
        if child_id not in self.children:
            self.children[child_id] = child
            if self.logger: self.logger.debug(f"Added child {child_id}")
        else:
            if self.logger: self.logger.debug(f"Child {child_id} already exists")

    def remove_child(self, child_id: Union[int, str]):
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
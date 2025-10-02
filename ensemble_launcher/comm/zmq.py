from typing import Union
import logging
import time
import socket
import random
import pickle
import sys
import os
from typing import TYPE_CHECKING
from .base import Comm
from .messages import Result

if TYPE_CHECKING:
    from ensemble_launcher.orchestrator.node import NodeInfo

try:
    import zmq
    ZMQ_AVAILABLE = True
except:
    ZMQ_AVAILABLE = False

logger = logging.getLogger(__name__)

class ZMQComm(Comm):
    def __init__(self, 
                 node_info: "NodeInfo",
                 parent_comm: "ZMQComm" = None,              
                 heartbeat_interval:int=1,
                 comm_config:dict={"comm_layer":"zmq"}):
        # Import NodeInfo here to avoid circular import
        from ensemble_launcher.orchestrator.node import NodeInfo
        if not ZMQ_AVAILABLE:
            logger.error(f"zmq is not available")
            raise ModuleNotFoundError
        
        self.node_info = node_info
        self.last_update_time = time.time()
        self.last_heartbeat_time = None
        self.heartbeat_interval = heartbeat_interval
        self._parent_comm = parent_comm

        self.comm_config = comm_config

        # ZMQ specific attributes
        self.parent_address = self._parent_comm.my_address if self._parent_comm is not None else None
        self.my_address = self.comm_config.get("parent-address",f"{socket.gethostname() if 'local' not in socket.gethostname() else 'localhost'}:5555")

        self.zmq_context = None
        self.router_socket = None
        self.dealer_socket = None
        self.router_cache = {}
        self.router_poller = None
        self.dealer_poller = None

    def setup_zmq_sockets(self):
        self.zmq_context = zmq.Context()
        if len(self.node_info.children_ids) > 0:
            self.router_socket = self.zmq_context.socket(zmq.ROUTER)
            self.router_socket.setsockopt(zmq.IDENTITY,f"{self.node_info.node_id}".encode())
            self.router_cache = {}
            try:
                self.router_socket.bind(f"tcp://{self.my_address}")
                logger.info(f"Successfully bound to {self.my_address}")
            except zmq.error.ZMQError as e:
                if "Address already in use" in str(e):
                    # Try binding up to 3 times with different ports
                    max_attempts = 3
                    for attempt in range(max_attempts):
                        try:
                            port = int(self.my_address.split(':')[-1]) + random.randint(1, 1000)
                            logger.info(f"Attempt {attempt+1}/{max_attempts}: Trying to bind to port {port} instead.")
                            self.my_address = f"{self.my_address.rsplit(':', 1)[0]}:{port}"
                            self.router_socket.bind(f"tcp://{self.my_address}")
                            logger.info(f"Successfully bound to {self.my_address}")
                            break  # Break out of the retry loop if binding succeeds
                        except zmq.error.ZMQError as retry_error:
                            if "Address already in use" in str(retry_error) and attempt < max_attempts - 1:
                                logger.warning(f"Port {port} also in use, retrying...")
                                continue
                            else:
                                raise retry_error
                else:
                    raise e
            self.router_poller = zmq.Poller()
            self.router_poller.register(self.router_socket, zmq.POLLIN)

        if self.parent_address is not None:
            self.dealer_socket = self.zmq_context.socket(zmq.DEALER)
            self.dealer_socket.setsockopt(zmq.IDENTITY,f"{self.node_info.node_id}".encode())
            logger.info(f"connecting to:{self.parent_address}")
            self.dealer_socket.connect(f"tcp://{self.parent_address}")
            self.dealer_poller = zmq.Poller()
            self.dealer_poller.register(self.dealer_socket, zmq.POLLIN)
            self.dealer_socket.send(pickle.dumps("READY"))
            msg = pickle.loads(self.dealer_socket.recv())
            if msg == "CONTINUE":
                logger.info(f"Received continue from parent")
            elif msg == "STOP":
                logger.info(f"Received stop from parent, Quitting...")
                sys.exit(0)
            else:
                if isinstance(msg, dict):
                    # Handle task update if needed
                    pass
                else:
                    logger.warning(f"Unexpected message from parent: {msg}. Expected dict or 'CONTINUE'/'STOP'.")

    def send_to_parent(self, parent_id: Union[int, str], data) -> int:
        if self.dealer_socket is not None:
            self.dealer_socket.send(pickle.dumps(data))
        else:
            logger.warning(f"Cannot send to parent {parent_id}: dealer_socket is not initialized.")
            return 1
        logger.debug(f"Sent message to parent {parent_id}: {data}")
        return 0

    def recv_from_parent(self, parent_id: Union[int, str], timeout: int = 60):
        if self.dealer_socket is not None:
            socks = dict(self.dealer_poller.poll(timeout * 1000))  # convert timeout to milliseconds
            if self.dealer_socket in socks and socks[self.dealer_socket] == zmq.POLLIN:
                msg = pickle.loads(self.dealer_socket.recv())
                logger.debug(f"Received message {msg} from parent {parent_id}.")
                return msg
        else:
            logger.warning(f"Cannot receive from parent {parent_id}: dealer_socket is not initialized.")
            raise RuntimeError("dealer_socket is not initialized")
        logger.debug(f"No message received from parent {parent_id} within timeout {timeout} seconds.")
        return None

    def send_to_child(self, child_id: Union[int, str], message) -> int:
        if child_id in self.node_info.children_ids:
            self.router_socket.send_multipart([f"{child_id}".encode(), pickle.dumps(message)])
            logger.debug(f"Sent message to child {child_id}")
            return 0
        else:
            return 1

    def recv_from_child(self, child_id: Union[int, str], timeout: int = 60):
        if child_id in self.node_info.children_ids:
            if child_id in self.router_cache and len(self.router_cache[child_id]) > 0:
                msg = self.router_cache[child_id].pop(0)  # Get the first cached message
                logger.debug(f"Received cached message from child {child_id}. {msg}")
                return msg
            tstart = time.time()
            while time.time() - tstart < timeout:
                socks = dict(self.router_poller.poll(100)) #wait for 100ms
                if self.router_socket in socks and socks[self.router_socket] == zmq.POLLIN:
                    msg = self.router_socket.recv_multipart()
                    msg[0] = msg[0].decode()  # Convert bytes to string for child_id
                    logger.debug(f"Received message from child {msg[0]} (expected {child_id})")
                    msg[1] = pickle.loads(msg[1])  # Unpickle the message
                    # wait for a message from the child
                    if msg[0] == str(child_id):
                        logger.debug(f"Received message {msg[1]} from child {child_id}.")
                        return msg[1]
                    else:
                        logger.debug(f"Received message from child {msg[0]}, but expected {child_id}. Caching the message.")
                        if msg[0] not in self.router_cache:
                            self.router_cache[msg[0]] = []
                        self.router_cache[msg[0]].append(msg[1])
        else:
            logger.debug(f"Cannot receive from child {child_id}: child does not exist.")
            raise ValueError(f"Child {child_id} does not exist.")
        logger.debug(f"No message received from child {child_id} within timeout {timeout} seconds.")
        return None

    def blocking_recv_from_parent(self, parent_id: Union[int, str]):
        """
        Blocking receive from a specific parent. Waits indefinitely until a message is available.
        """
        logger.debug(f"Waiting for message from parent {parent_id}......")
        msg = pickle.loads(self.dealer_socket.recv())  # Blocking call
        logger.debug(f"Received message from parent {parent_id} (blocking)")
        return msg

    def blocking_recv_from_child(self, child_id: Union[int, str]):
        """
        Blocking receive from a specific child. Waits indefinitely until a message is available.
        """
        if child_id in self.node_info.children_ids:
            logger.debug(f"Waiting for message from child {child_id}......")
            if child_id in self.router_cache and len(self.router_cache[child_id]) > 0:
                logger.debug(f"Child {child_id} has cached messages.")
                msg = self.router_cache[child_id].pop(0)  # Get the first cached message
            while True:
                msgs = self.router_socket.recv_multipart()  # Blocking call
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
            logger.debug(f"Cannot receive from child {child_id}: child does not exist.")
            raise ValueError(f"Child {child_id} does not exist.")
        logger.debug(f"Received message {msg} from child {child_id}")
        return msg
    
    def return_result(self, result):
        """Return result to parent"""
        if self.dealer_socket is not None:
            self.dealer_socket.send(pickle.dumps(Result(result)))
        else:
            logger.warning("Cannot return result: dealer_socket is not initialized.")
    
    def get_results(self, timeout: float = None):
        results = []
        if timeout is not None:
            wait_time = timeout/len(self.node_info.children_ids)
        else:
            wait_time = 0.1

        for child_id in self.node_info.children_ids:
            msg = self.recv_from_child(child_id=child_id, timeout=wait_time)
            if isinstance(msg, Result):
                results.append(msg.data)
            else:
                if child_id in self.router_cache:
                    self.router_cache[child_id].append(msg)
                else:
                    self.router_cache[child_id] = [msg]
        
        # drain any cached Result messages and append to results
        for child_id, cached_msgs in list(self.router_cache.items()):
            remaining_msgs = []
            for msg in cached_msgs:
                if isinstance(msg, Result):
                    results.append(msg.data)
                else:
                    remaining_msgs.append(msg)
            self.router_cache[child_id] = remaining_msgs

        return results
            
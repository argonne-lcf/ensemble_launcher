from typing import Union
import logging
import time
import socket
import random
import pickle
import sys
from .base import Comm, NodeInfo
from .messages import Result
from typing import Any, Optional
import cloudpickle


try:
    import zmq
    ZMQ_AVAILABLE = True
except:
    ZMQ_AVAILABLE = False

logger = logging.getLogger(__name__)

class ZMQComm(Comm):
    def __init__(self, 
                 node_info: NodeInfo,
                 parent_comm: "ZMQComm" = None,              
                 heartbeat_interval:int=1,
                 parent_address: str = None ###parent comm is not always pickleble
                 ):
        if not ZMQ_AVAILABLE:
            logger.error(f"zmq is not available")
            raise ModuleNotFoundError
        
        super().__init__(node_info,parent_comm,heartbeat_interval)

        # ZMQ specific attributes
        self.parent_address = self._parent_comm.my_address if self._parent_comm is not None else parent_address
        self.my_address = f"{socket.gethostname() if 'local' not in socket.gethostname() else 'localhost'}:5555"

        self.zmq_context = None
        self.router_socket = None
        self.dealer_socket = None
        self.router_poller = None
        self.dealer_poller = None

        # self.setup_zmq_sockets()
        # self.send_heartbeat()

    def setup_zmq_sockets(self):
        self.zmq_context = zmq.Context()
        if len(self.node_info.children_ids) > 0:
            self.router_socket = self.zmq_context.socket(zmq.ROUTER)
            self.router_socket.setsockopt(zmq.IDENTITY,f"{self.node_info.node_id}".encode())
            try:
                self.router_socket.bind(f"tcp://{self.my_address}")
                logger.info(f"{self.node_info.node_id}: Successfully bound to {self.my_address}")
            except zmq.error.ZMQError as e:
                if "Address already in use" in str(e):
                    # Try binding up to 3 times with different ports
                    max_attempts = 10
                    for attempt in range(max_attempts):
                        try:
                            port = int(self.my_address.split(':')[-1]) + random.randint(1, 1000)
                            logger.info(f"{self.node_info.node_id}: Attempt {attempt+1}/{max_attempts}: Trying to bind to port {port} instead.")
                            self.my_address = f"{self.my_address.rsplit(':', 1)[0]}:{port}"
                            self.router_socket.bind(f"tcp://{self.my_address}")
                            logger.info(f"{self.node_info.node_id}: Successfully bound to {self.my_address}")
                            break  # Break out of the retry loop if binding succeeds
                        except zmq.error.ZMQError as retry_error:
                            if "Address already in use" in str(retry_error) and attempt < max_attempts - 1:
                                logger.warning(f"{self.node_info.node_id}: Port {port} also in use, retrying...")
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
            # logger.info(f"{self.node_info.node_id}: connecting to:{self.parent_address}")
            self.dealer_socket.connect(f"tcp://{self.parent_address}")
            self.dealer_poller = zmq.Poller()
            self.dealer_poller.register(self.dealer_socket, zmq.POLLIN)
            logger.info(f"{self.node_info.node_id}: connected to:{self.parent_address}")
            time.sleep(1.0)

    def _send_to_parent(self, data: Any) -> bool:
        if self.node_info.parent_id is None:
            logger.warning(f"{self.node_info.node_id}: No parent connection available")
            return False
        
        try:
            self.dealer_socket.send(cloudpickle.dumps(data))
            logger.debug(f"{self.node_info.node_id}: Sent message to parent: {data}")
            return True
        except Exception as e:
            logger.warning(f"{self.node_info.node_id}: Sending message to parent failed with {e}")
            return False

    def _recv_from_parent(self, timeout: Optional[float] = None) -> Any:
        if self.node_info.parent_id is None:
            logger.warning(f"{self.node_info.node_id}: No parent connection available")
            return None
        
        try:
            socks = dict(self.dealer_poller.poll((timeout * 1000) if timeout is not None else None))  # convert timeout to milliseconds
            if self.dealer_socket in socks and socks[self.dealer_socket] == zmq.POLLIN:
                msg = cloudpickle.loads(self.dealer_socket.recv())
                logger.debug(f"{self.node_info.node_id}: Received message {msg} from parent.")
                return msg
            logger.debug(f"{self.node_info.node_id}: No message received from parent within timeout {timeout} seconds.")
            return None
        except Exception as e:
            logger.warning(f"{self.node_info.node_id}: Receiving message failed with exception {e}!")  # Fixed typo
            return None

    def _send_to_child(self, child_id: str, data: Any) -> bool:
        if child_id not in self.node_info.children_ids:
            logger.warning(f"{self.node_info.node_id}: No connection to child {child_id}")
            return False
        
        try:
            self.router_socket.send_multipart([f"{child_id}".encode(), cloudpickle.dumps(data)])
            logger.debug(f"{self.node_info.node_id}: Sent message to child {child_id}")
            return True
        except Exception as e:
            logger.warning(f"{self.node_info.node_id}: Sending message to child {child_id} failed with {e}")
            return False

    def _recv_from_child(self, child_id: str, timeout: Optional[float] = None) -> Any:
        if child_id not in self.node_info.children_ids:
            logger.warning(f"{self.node_info.node_id}: No connection to child {child_id}")
            return None
        
        try:
            if child_id in self._cache and len(self._cache[child_id]) > 0:
                msg = self._cache[child_id].pop(0)  # Get the first cached message
                logger.debug(f"{self.node_info.node_id}: Received (cached) message from child {child_id}.")
                return msg
            
            start_time = time.time()
            while True:

                if timeout is not None and time.time() - start_time >= timeout:
                    break

                socks = dict(self.router_poller.poll(100)) #wait for 100ms
                if self.router_socket in socks and socks[self.router_socket] == zmq.POLLIN:
                    msg = self.router_socket.recv_multipart()
                    msg[0] = msg[0].decode()  # Convert bytes to string for child_id
                    msg[1] = cloudpickle.loads(msg[1])  # Unpickle the message
                    # wait for a message from the child
                    if msg[0] == str(child_id):
                        logger.debug(f"{self.node_info.node_id}: Received message {msg} from child {child_id}.")
                        return msg[1]
                    else:
                        self._cache[msg[0]].append(msg[1])
            logger.debug(f"{self.node_info.node_id}: No message received from child {child_id} within timeout {timeout} seconds.")
            return None
        except Exception as e:
            logger.warning(f"{self.node_info.node_id}: Receiving message from child {child_id} failed with exception {e}!")
            return None
    

    def close(self):
        """Clean up ZMQ resources."""
        try:
            if self.router_socket:
                self.router_socket.close()
            if self.dealer_socket:
                self.dealer_socket.close()
            if self.zmq_context:
                self.zmq_context.term()
        except Exception as e:
            logger.warning(f"{self.node_info.node_id}: Error during ZMQ cleanup: {e}")
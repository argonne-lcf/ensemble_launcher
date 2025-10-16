from typing import Dict
import time
import socket
import random
from .base import Comm, NodeInfo
from typing import Any, Optional
import cloudpickle
from logging import Logger
import queue


try:
    import zmq
    ZMQ_AVAILABLE = True
except:
    ZMQ_AVAILABLE = False

# logger = logging.getLogger(__name__)

class ZMQComm(Comm):
    def __init__(self, 
                 logger: Logger,
                 node_info: NodeInfo,
                 parent_comm: "ZMQComm" = None,              
                 heartbeat_interval:int=1,
                 parent_address: str = None, ###parent comm is not always pickleble
                 profile: bool = True
                 ):
        
        super().__init__(logger, node_info,parent_comm,heartbeat_interval,profile=profile)
        if not ZMQ_AVAILABLE:
            self.logger.error(f"zmq is not available")
            raise ModuleNotFoundError

        # ZMQ specific attributes
        self.parent_address = self._parent_comm.my_address if self._parent_comm is not None else parent_address
        self.my_address = f"{socket.gethostname() if 'local' not in socket.gethostname() else 'localhost'}:{5555+random.randint(1, 1000)}"

        self.zmq_context = None
        self.router_socket = None
        self.dealer_socket = None
        self.router_poller = None
        self.dealer_poller = None
        
        self._router_cache = None

    
    def _init_router_cache(self):
        # ZMQ-specific raw data cache using FIFO queues (preserves message order)
        self._router_cache: Dict[str, queue.Queue] = {}
        for child_id in self.node_info.children_ids:
            self._router_cache[child_id] = queue.Queue()
        
        if self.node_info.parent_id:
            self._router_cache[self.node_info.parent_id] = queue.Queue()

    def setup_zmq_sockets(self):
        if not self._router_cache:
            self._init_router_cache()

        self.zmq_context = zmq.Context()
        if len(self.node_info.children_ids) > 0:
            self.router_socket = self.zmq_context.socket(zmq.ROUTER)
            self.router_socket.setsockopt(zmq.IDENTITY,f"{self.node_info.node_id}".encode())
            try:
                self.router_socket.bind(f"tcp://{self.my_address}")
                self.logger.info(f"{self.node_info.node_id}: Successfully bound to {self.my_address}")
            except zmq.error.ZMQError as e:
                if "Address already in use" in str(e):
                    # Try binding up to 3 times with different ports
                    max_attempts = 10
                    for attempt in range(max_attempts):
                        try:
                            port = int(self.my_address.split(':')[-1]) + random.randint(1, 1000)
                            self.logger.info(f"{self.node_info.node_id}: Attempt {attempt+1}/{max_attempts}: Trying to bind to port {port} instead.")
                            self.my_address = f"{self.my_address.rsplit(':', 1)[0]}:{port}"
                            self.router_socket.bind(f"tcp://{self.my_address}")
                            self.logger.info(f"{self.node_info.node_id}: Successfully bound to {self.my_address}")
                            break  # Break out of the retry loop if binding succeeds
                        except zmq.error.ZMQError as retry_error:
                            if "Address already in use" in str(retry_error) and attempt < max_attempts - 1:
                                self.logger.warning(f"{self.node_info.node_id}: Port {port} also in use, retrying...")
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
            # self.logger.info(f"{self.node_info.node_id}: connecting to:{self.parent_address}")
            self.dealer_socket.connect(f"tcp://{self.parent_address}")
            self.dealer_poller = zmq.Poller()
            self.dealer_poller.register(self.dealer_socket, zmq.POLLIN)
            self.logger.info(f"{self.node_info.node_id}: connected to:{self.parent_address}")
            time.sleep(1.0)

    def _send_to_parent(self, data: Any) -> bool:
        if self.node_info.parent_id is None:
            self.logger.warning(f"{self.node_info.node_id}: No parent connection available")
            return False
        
        try:
            self.dealer_socket.send(cloudpickle.dumps(data))
            self.logger.debug(f"{self.node_info.node_id}: Sent message to parent: {data}")
            return True
        except Exception as e:
            self.logger.warning(f"{self.node_info.node_id}: Sending message to parent failed with {e}")
            return False

    def _recv_from_parent(self, timeout: Optional[float] = None) -> Any:
        if self.node_info.parent_id is None:
            self.logger.warning(f"{self.node_info.node_id}: No parent connection available")
            return None
        
        try:
            # Check ZMQ-specific FIFO cache first for raw data
            parent_id = self.node_info.parent_id
            if parent_id in self._router_cache:
                try:
                    raw_data = self._router_cache[parent_id].get_nowait()
                    self.logger.debug(f"{self.node_info.node_id}: Received (cached) raw data from parent.")
                    return raw_data
                except queue.Empty:
                    pass
            
            socks = dict(self.dealer_poller.poll((timeout * 1000) if timeout is not None else None))  # convert timeout to milliseconds
            if self.dealer_socket in socks and socks[self.dealer_socket] == zmq.POLLIN:
                raw_data = cloudpickle.loads(self.dealer_socket.recv())
                self.logger.debug(f"{self.node_info.node_id}: Received raw data from parent.")
                return raw_data
            self.logger.debug(f"{self.node_info.node_id}: No message received from parent within timeout {timeout} seconds.")
            return None
        except Exception as e:
            self.logger.warning(f"{self.node_info.node_id}: Receiving message failed with exception {e}!")
            return None

    def _send_to_child(self, child_id: str, data: Any) -> bool:
        if child_id not in self.node_info.children_ids:
            self.logger.warning(f"{self.node_info.node_id}: No connection to child {child_id}")
            return False
        
        try:
            self.router_socket.send_multipart([f"{child_id}".encode(), cloudpickle.dumps(data)])
            self.logger.debug(f"{self.node_info.node_id}: Sent message to child {child_id}")
            return True
        except Exception as e:
            self.logger.warning(f"{self.node_info.node_id}: Sending message to child {child_id} failed with {e}")
            return False

    def _recv_from_child(self, child_id: str, timeout: Optional[float] = None) -> Any:
        if child_id not in self.node_info.children_ids:
            self.logger.warning(f"{self.node_info.node_id}: No connection to child {child_id}")
            return None
        
        try:
            # Check ZMQ-specific FIFO cache first for raw data
            if child_id in self._router_cache:
                try:
                    raw_data = self._router_cache[child_id].get_nowait()
                    self.logger.debug(f"{self.node_info.node_id}: Received (cached) raw data from child {child_id}.")
                    return raw_data
                except queue.Empty:
                    pass
            
            start_time = time.time()
            while True:
                if timeout is not None and time.time() - start_time >= timeout:
                    break

                socks = dict(self.router_poller.poll(100)) #wait for 100ms
                if self.router_socket in socks and socks[self.router_socket] == zmq.POLLIN:
                    msg = self.router_socket.recv_multipart()
                    sender_id = msg[0].decode()  # Convert bytes to string for child_id
                    raw_data = cloudpickle.loads(msg[1])  # Unpickle the raw data
                    
                    # Check if this is the message we're waiting for
                    if sender_id == str(child_id):
                        self.logger.debug(f"{self.node_info.node_id}: Received raw data from child {child_id}.")
                        return raw_data
                    else:
                        # Cache raw data for other children in ZMQ-specific FIFO cache
                        if sender_id in self._router_cache:
                            self._router_cache[sender_id].put(raw_data)
                        else:
                            self.logger.warning(f"{self.node_info.node_id}: Received data from unknown child {sender_id}")
            
            self.logger.debug(f"{self.node_info.node_id}: No message received from child {child_id} within timeout {timeout} seconds.")
            return None
        except Exception as e:
            self.logger.warning(f"{self.node_info.node_id}: Receiving message from child {child_id} failed with exception {e}!")
            return None

    def update_node_info(self, node_info: NodeInfo):
        """Override to update ZMQ-specific cache as well"""
        super().update_node_info(node_info)
        
        # Update ZMQ-specific FIFO cache
        for child_id in self.node_info.children_ids:
            if child_id not in self._router_cache:
                self._router_cache[child_id] = queue.Queue()

    def close(self):
        """Clean up ZMQ resources."""
        try:
            # Clear ZMQ-specific FIFO cache
            for cache_queue in self._router_cache.values():
                try:
                    while True:
                        cache_queue.get_nowait()
                except queue.Empty:
                    pass
            self._router_cache.clear()
            
            # Close ZMQ resources
            if self.router_socket:
                self.router_socket.close()
            if self.dealer_socket:
                self.dealer_socket.close()
            if self.zmq_context:
                self.zmq_context.term()
        except Exception as e:
            self.logger.warning(f"{self.node_info.node_id}: Error during ZMQ cleanup: {e}")
    
    def pickable_copy(self):
        ret = ZMQComm(None, node_info=self.node_info, parent_address=self.parent_address)
        ret.my_address = self.my_address
        return ret
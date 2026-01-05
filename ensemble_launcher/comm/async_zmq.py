from typing import Dict
import time
import socket
import random

from .async_base import AsyncComm
from .nodeinfo import NodeInfo
from typing import Any, Optional
import cloudpickle
from logging import Logger
from dataclasses import asdict
import asyncio
from asyncio import Queue


try:    
    import zmq
    from zmq.asyncio import Poller, Context, Socket
    ZMQ_AVAILABLE = True
except:
    ZMQ_AVAILABLE = False

# logger = logging.getLogger(__name__)

class AsyncZMQComm(AsyncComm):
    def __init__(self, 
                 logger: Logger,
                 node_info: NodeInfo,
                 parent_comm: "AsyncZMQComm" = None,              
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
        
        self._router_cache = None

        self._stop_event = None
        
    async def init_cache(self):
        await super().init_cache()
        await self._init_router_cache()
        
    async def _init_router_cache(self):
        if self._router_cache is None:
            self._router_cache: Dict[str, Queue] = {}
            
        for child_id in self._node_info.children_ids:
            if child_id not in self._router_cache:
                self._router_cache[child_id] = Queue()
        
        if self._node_info.parent_id:
            if self._node_info.parent_id not in self._router_cache:
                self._router_cache[self._node_info.parent_id] = Queue()

    async def setup_zmq_sockets(self):
        if not self._router_cache:
            await self._init_router_cache()

        self.zmq_context = Context()
        self.router_socket = self.zmq_context.socket(zmq.ROUTER,socket_class=Socket)
        self.router_socket.setsockopt(zmq.IDENTITY,f"{self._node_info.node_id}".encode())
        try:
            self.router_socket.bind(f"tcp://{self.my_address}")
            self.logger.info(f"{self._node_info.node_id}: Successfully bound to {self.my_address}")
        except zmq.error.ZMQError as e:
            if "Address already in use" in str(e):
                # Try binding up to 3 times with different ports
                max_attempts = 10
                for attempt in range(max_attempts):
                    try:
                        port = int(self.my_address.split(':')[-1]) + random.randint(1, 1000)
                        self.logger.info(f"{self._node_info.node_id}: Attempt {attempt+1}/{max_attempts}: Trying to bind to port {port} instead.")
                        self.my_address = f"{self.my_address.rsplit(':', 1)[0]}:{port}"
                        self.router_socket.bind(f"tcp://{self.my_address}")
                        self.logger.info(f"{self._node_info.node_id}: Successfully bound to {self.my_address}")
                        break  # Break out of the retry loop if binding succeeds
                    except zmq.error.ZMQError as retry_error:
                        if "Address already in use" in str(retry_error) and attempt < max_attempts - 1:
                            self.logger.warning(f"{self._node_info.node_id}: Port {port} also in use, retrying...")
                            continue
                        else:
                            raise retry_error
            else:
                raise e

        if self.parent_address is not None:
            self.dealer_socket = self.zmq_context.socket(zmq.DEALER,socket_class=Socket)
            self.dealer_socket.setsockopt(zmq.IDENTITY,f"{self._node_info.node_id}".encode())
            self.dealer_socket.connect(f"tcp://{self.parent_address}")
            self.logger.info(f"{self._node_info.node_id}: connected to:{self.parent_address}")

    async def start_monitors(self,**kwargs):
        """Start background tasks to monitor ZMQ sockets."""
        await super().start_monitors(**kwargs)
        if self._node_info.parent_id is not None:
            asyncio.create_task(self._monitor_parent_socket())
        if len(self._node_info.children_ids) > 0:
            asyncio.create_task(self._monitor_child_sockets())
        
    async def _monitor_parent_socket(self) -> None:
        """
        Monitor the parent dealer socket for incoming messages and cache them.
        """
        await self._init_router_cache()
        # Check ZMQ-specific FIFO cache first for raw data
        parent_id = self._node_info.parent_id
        failures = 0
        while not self._stop_event.is_set():
            try:
                raw_data = await self.dealer_socket.recv()
                await self._router_cache[parent_id].put(cloudpickle.loads(raw_data))
                self.logger.debug(f"{self._node_info.node_id}: Cached raw data from parent.")
            except Exception as e:
                failures += 1
                self.logger.warning(f"{self._node_info.node_id}: Error caching data from parent failed {failures} times: {e}")
                await asyncio.sleep(0.1)  # Avoid busy waiting

    async def _monitor_child_sockets(self) -> None:
        """
        Monitor the child router sockets for incoming messages and cache them.
        """
        await self._init_router_cache()
        failures = 0
        while not self._stop_event.is_set():
            try:
                raw_data = await self.router_socket.recv_multipart()
                sender_id = raw_data[0].decode()  # Convert bytes to string for child_id
                data = cloudpickle.loads(raw_data[1])  # Unpickle the raw data
                await self._router_cache[sender_id].put(data)
            except Exception as e:
                failures += 1
                self.logger.warning(f"{self._node_info.node_id}: Error caching data from child failed {failures} times: {e}")
                await asyncio.sleep(0.1)
            

    async def _send_to_parent(self, data: Any) -> bool:
        if self._node_info.parent_id is None:
            self.logger.warning(f"{self._node_info.node_id}: No parent connection available to {self._node_info.parent_id}")
            return False
        
        try:
            self.dealer_socket.send(cloudpickle.dumps(data))
            self.logger.debug(f"{self._node_info.node_id}: Sent message to parent: {data}")
            return True
        except Exception as e:
            self.logger.warning(f"{self._node_info.node_id}: Sending message to parent failed with {e}")
            return False

    async def _recv_from_parent(self, timeout: Optional[float] = None) -> Any:
        if self._node_info.parent_id is None:
            self.logger.error(f"{self._node_info.node_id}: No parent connection available")
            raise RuntimeError("No parent connection available")
        
        try:
            # Check ZMQ-specific FIFO cache first for raw data
            parent_id = self._node_info.parent_id
            self.logger.debug(f"{self._node_info.node_id}: Waiting to receive message from parent {parent_id} with timeout {timeout}")
            return await asyncio.wait_for(self._router_cache[parent_id].get(), timeout=timeout)
        except asyncio.TimeoutError:
            self.logger.debug(f"{self._node_info.node_id}: No message received from parent within timeout {timeout} seconds.")
            return None
        except Exception as e:
            self.logger.warning(f"{self._node_info.node_id}: Receiving message from parent failed with exception {e}!")
            return None

    async def _send_to_child(self, child_id: str, data: Any) -> bool:
        if child_id not in self._node_info.children_ids:
            self.logger.error(f"{self._node_info.node_id}: No connection to child {child_id}")
            raise RuntimeError(f"No connection to child {child_id}")
        
        try:
            self.router_socket.send_multipart([f"{child_id}".encode(), cloudpickle.dumps(data)])
            self.logger.debug(f"{self._node_info.node_id}: Sent message to child {child_id}")
            return True
        except Exception as e:
            self.logger.warning(f"{self._node_info.node_id}: Sending message to child {child_id} failed with {e}")
            return False

    async def _recv_from_child(self, child_id: str, timeout: Optional[float] = None) -> Any:
        if child_id not in self._node_info.children_ids:
            self.logger.error(f"{self._node_info.node_id}: No connection to child {child_id}")
            raise RuntimeError(f"No connection to child {child_id}")
        
        try:
            return await asyncio.wait_for(self._router_cache[child_id].get(), timeout=timeout)
        except asyncio.TimeoutError:
            self.logger.debug(f"{self._node_info.node_id}: No message received from child {child_id} within timeout {timeout} seconds.")
            return None
        except Exception as e:
            self.logger.warning(f"{self._node_info.node_id}: Receiving message from child {child_id} failed with exception {e}!")
            return None

    async def close(self):
        """Clean up ZMQ resources."""
        self._stop_event.set() ##signal monitors to stop
        await super().clear_cache()
        try:
            # Clear ZMQ-specific FIFO cache
            for cache_queue in self._router_cache.values():
                try:
                    while True:
                        cache_queue.get_nowait()
                except asyncio.QueueEmpty:
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
            self.logger.warning(f"{self._node_info.node_id}: Error during ZMQ cleanup: {e}")
    
    async def pickable_copy(self) -> "AsyncZMQComm":
        ret = AsyncZMQComm(None, node_info=self._node_info, parent_address=self.parent_address)
        ret.my_address = self.my_address
        return ret
    
    async def asdict(self):
        base_dict = {}
        base_dict["node_info"] = asdict(self._node_info) if self._node_info else None
        base_dict["parent_address"] = self.parent_address
        base_dict["my_address"] = self.my_address
        return base_dict
    
    @classmethod
    async def fromdict(cls, data: Dict[str, Any]) -> "AsyncZMQComm":
        node_info = NodeInfo(**data["node_info"]) if data.get("node_info") else None
        comm = cls(None, node_info=node_info, parent_address=data.get("parent_address"))
        comm.my_address = data.get("my_address")
        return comm
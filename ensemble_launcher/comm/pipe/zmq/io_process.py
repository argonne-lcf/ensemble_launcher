"""The ZMQ IOProcesses: ROUTER / DEALER sockets, their options and frame layouts."""

from typing import List, Optional, Union

from ..io_process import IOProcess


class ZMQRouterIOProcess(IOProcess):
    """The ROUTER socket: options, bind (with retry on a taken port) and frame layout."""

    server = True

    def __init__(self, *args, address: str, lightweight: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._address = address
        self._lightweight = lightweight
        self._context = None
        self._socket = None

    def kwargs(self):
        return {**super().kwargs(), "address": self._address, "lightweight": self._lightweight}

    @property
    def address(self) -> str:
        return self._address

    async def _raw_open(self) -> None:
        import random

        import zmq
        from zmq.asyncio import Context, Socket

        self._context = Context()
        self._socket = self._context.socket(zmq.ROUTER, socket_class=Socket)
        self._socket.setsockopt(
            zmq.IDENTITY, f"{self._identity}:{self._secret_id}".encode()
        )

        if self._lightweight:
            self._context.set(zmq.IO_THREADS, 1)
            self._socket.setsockopt(zmq.SNDHWM, 100)
            self._socket.setsockopt(zmq.RCVHWM, 100)
            self._socket.setsockopt(zmq.SNDTIMEO, 100)
        else:
            self._context.set(zmq.IO_THREADS, 4)
            self._socket.setsockopt(zmq.SNDHWM, 10000)
            self._socket.setsockopt(zmq.RCVHWM, 10000)
            self._socket.setsockopt(zmq.HEARTBEAT_IVL, 5000)
            self._socket.setsockopt(zmq.HEARTBEAT_TIMEOUT, 15000)
            self._socket.setsockopt(zmq.HEARTBEAT_TTL, 15000)
            self._socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
            self._socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
            self._socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 10)
            self._socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 5)
            self._socket.setsockopt(zmq.TCP_KEEPALIVE_CNT, 3)

        try:
            self._socket.bind(f"tcp://{self._address}")
        except zmq.error.ZMQError as e:
            if "Address already in use" in str(e):
                max_attempts = 10
                for attempt in range(max_attempts):
                    try:
                        port = int(self._address.split(":")[-1]) + random.randint(
                            1, 1000
                        )
                        self._address = f"{self._address.rsplit(':', 1)[0]}:{port}"
                        self._socket.bind(f"tcp://{self._address}")
                        break
                    except zmq.error.ZMQError as retry_error:
                        if (
                            "Address already in use" in str(retry_error)
                            and attempt < max_attempts - 1
                        ):
                            continue
                        else:
                            raise retry_error
            else:
                raise e
        endpoint = self._socket.getsockopt(zmq.LAST_ENDPOINT).decode()
        self._address = endpoint.replace("tcp://", "")
        self.logger.info(f"Server bound to {self._address}")

    async def _raw_close(self) -> None:
        import zmq

        if self._socket:
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.close()
            self._socket = None
        if self._context:
            self._context.term()
            self._context = None

    async def _raw_send(
        self,
        data: Union[bytes, List[bytes]],
        msg_id: Optional[int] = None,
        target_id: Optional[str] = None,
    ) -> bool:
        """Send frames via ZMQ Router.

        Sends: [target_id, identity_frame, msg_id(8B)?, *data_frames]
        """
        identity_frame = f"{self._identity}:{self._secret_id}".encode()
        if msg_id is not None:
            frames = [
                target_id.encode(),
                identity_frame,
                msg_id.to_bytes(8, "big"),
            ]
        else:
            frames = [target_id.encode(), identity_frame]
        if isinstance(data, list):
            frames.extend(data)
        else:
            frames.append(data)
        try:
            await self._socket.send_multipart(frames)
        except Exception:
            if self._lightweight:
                return False
            raise
        return True

    async def _raw_recv(self) -> List[bytes]:
        """Receive frames via ZMQ Router.

        Returns: [dealer_routing_id, msg_id(8B)?, *data_frames]
        ZMQ prepends the dealer's routing id automatically.
        """
        return await self._socket.recv_multipart()


class ZMQDealerIOProcess(IOProcess):
    """The DEALER socket: options, connect and frame layout."""

    server = False

    def __init__(self, *args, remote_address: str, lightweight: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._remote_address = remote_address
        self._lightweight = lightweight
        self._context = None
        self._socket = None

    def kwargs(self):
        return {
            **super().kwargs(),
            "remote_address": self._remote_address,
            "lightweight": self._lightweight,
        }

    @property
    def address(self) -> Optional[str]:
        return None

    async def _raw_open(self) -> None:
        import zmq
        from zmq.asyncio import Context, Socket

        self._context = Context()
        self._socket = self._context.socket(zmq.DEALER, socket_class=Socket)
        self._socket.setsockopt(
            zmq.IDENTITY, f"{self._identity}:{self._secret_id}".encode()
        )

        if self._lightweight:
            self._context.set(zmq.IO_THREADS, 1)
            self._socket.setsockopt(zmq.SNDHWM, 100)
            self._socket.setsockopt(zmq.RCVHWM, 100)
        else:
            self._context.set(zmq.IO_THREADS, 4)
            self._socket.setsockopt(zmq.SNDHWM, 10000)
            self._socket.setsockopt(zmq.RCVHWM, 10000)
            self._socket.setsockopt(zmq.HEARTBEAT_IVL, 5000)
            self._socket.setsockopt(zmq.HEARTBEAT_TIMEOUT, 15000)
            self._socket.setsockopt(zmq.HEARTBEAT_TTL, 15000)
            self._socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
            self._socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 10)
            self._socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 5)
            self._socket.setsockopt(zmq.TCP_KEEPALIVE_CNT, 3)

        self._socket.connect(f"tcp://{self._remote_address}")
        self.logger.info(f"Connected to {self._remote_address}")

    async def _raw_close(self) -> None:
        import zmq

        if self._socket:
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.close()
            self._socket = None
        if self._context:
            self._context.term()
            self._context = None

    async def _raw_send(
        self,
        data: Union[bytes, List[bytes]],
        msg_id: Optional[int] = None,
        target_id: Optional[str] = None,
    ) -> bool:
        """Send frames via ZMQ Dealer.

        Sends: [msg_id(8B)?, *data_frames]
        """
        if msg_id is not None:
            frames = [msg_id.to_bytes(8, "big")]
        else:
            frames = []
        if isinstance(data, list):
            frames.extend(data)
        else:
            frames.append(data)
        await self._socket.send_multipart(frames)
        return True

    async def _raw_recv(self) -> List[bytes]:
        """Receive frames via ZMQ Dealer.

        Returns: [identity_frame, msg_id(8B)?, *data_frames]
        ZMQ strips the routing id; the first frame is the sender's identity.
        """
        return await self._socket.recv_multipart()



__all__ = ["ZMQRouterIOProcess", "ZMQDealerIOProcess"]

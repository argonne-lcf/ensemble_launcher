import re
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel


from .async_connection import (
    ClientConnection,
    ClientConnectionState,
    ServerConnection,
    ServerConnectionState,
)

T = TypeVar("T", bound="AsyncTransportState")

def get_hsn_ip_cli(ifname="hsn0")->Optional[str]:
    """
    Retrieves the IPv4 address by parsing the standard Linux `ip` command.
    """
    try:
        # Run: ip -4 addr show hsn0
        output = subprocess.check_output(
            ["ip", "-4", "addr", "show", ifname],
            stderr=subprocess.DEVNULL
        ).decode('utf-8')

        # Regex to match the first IPv4 address in the output
        match = re.search(r'inet\s+(\d+\.\d+\.\d+\.\d+)', output)
        if match:
            return match.group(1)

    except Exception:
        pass # Interface doesn't exist or command failed




class AsyncTransportState(BaseModel):
    transport_type: str
    server_connections: Dict[str, ServerConnectionState] = {}
    client_connections: Dict[str, ClientConnectionState] = {}

    def serialize(self, *args, **kwargs) -> str:
        return self.model_dump_json(*args, **kwargs)

    @classmethod
    def deserialize(cls: Type[T], data: str) -> T:
        return cls.model_validate_json(data)


class AsyncTransport(ABC):
    transport_type: str = ""

    def __init__(self):
        self._server_connections: Dict[str, ServerConnection] = {}
        self._client_connections: Dict[str, ClientConnection] = {}

    def get_server_connection(
        self, identity: str, secret_id: str, req_res: bool = False, **kwargs
    ) -> Optional[ServerConnection]:
        key = f"{identity}:{secret_id}"
        conn = self._server_connections.get(key)
        if conn is None:
            conn = self._create_server_connection(
                identity, secret_id, req_res=req_res, **kwargs
            )
            self._server_connections[key] = conn
        return conn

    def get_client_connection(
        self, identity: str, secret_id: str, req_res: bool = False, **kwargs
    ) -> Optional[ClientConnection]:
        key = f"{identity}:{secret_id}"
        conn = self._client_connections.get(key)
        if conn is None and kwargs:
            conn = self._create_client_connection(
                identity, secret_id, req_res=req_res, **kwargs
            )
            self._client_connections[key] = conn
        return conn

    def get_server_connections(self) -> List[ServerConnection]:
        seen = set()
        result = []
        for conn in self._server_connections.values():
            if id(conn) not in seen:
                seen.add(id(conn))
                result.append(conn)
        return result

    def get_client_connections(self) -> List[ClientConnection]:
        return list(self._client_connections.values())

    @abstractmethod
    def _create_server_connection(
        self, identity: str, secret_id: str, req_res: bool = False, **kwargs
    ) -> ServerConnection:
        pass

    @abstractmethod
    def _create_client_connection(
        self, identity: str, secret_id: str, req_res: bool = False, **kwargs
    ) -> ClientConnection:
        pass

    @abstractmethod
    def create_child_pipe(
        self,
        parent_id: str,
        parent_secret: str,
        child_id: str,
        child_secret: str,
        req_res: bool = False,
    ) -> Tuple[ServerConnection, ClientConnection]:
        pass

    @abstractmethod
    def get_state(self) -> Optional[AsyncTransportState]:
        pass

    @classmethod
    @abstractmethod
    def set_state(cls, state: AsyncTransportState) -> "AsyncTransport":
        pass

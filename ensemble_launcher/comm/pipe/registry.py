from typing import Any, Dict, Optional, Type


class TransportRegistry:
    def __init__(self):
        self._entries: Dict[str, Dict[str, Optional[Type]]] = {}

    def register(
        self,
        name: str,
        *,
        transport_state: Optional[Type] = None,
        server_connection: Type,
        server_connection_state: Optional[Type] = None,
        client_connection: Type,
        client_connection_state: Optional[Type] = None,
    ):
        def decorator(cls: Type[Any]) -> Type[Any]:
            self._entries[name] = {
                "transport": cls,
                "transport_state": transport_state,
                "server_connection": server_connection,
                "server_connection_state": server_connection_state,
                "client_connection": client_connection,
                "client_connection_state": client_connection_state,
            }
            return cls

        return decorator

    def get(self, name: str) -> Optional[Dict[str, Optional[Type]]]:
        return self._entries.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __getitem__(self, name: str) -> Dict[str, Optional[Type]]:
        return self._entries[name]

    @property
    def available(self):
        return list(self._entries.keys())


transport_registry = TransportRegistry()

import random
import socket
from typing import Optional, Tuple

def find_free_port(
    port_range: Tuple[int, int], host: str = "127.0.0.1"
) -> Optional[int]:
    """
    Attempts to find a free port within the given range by binding to it.
    Checks ports in a random order to reduce collisions between concurrent startups.
    """
    # Create a list of all ports in the range and shuffle them
    ports_to_check = list(range(port_range[0], port_range[1]))
    random.shuffle(ports_to_check)

    for port in ports_to_check:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue

    return None

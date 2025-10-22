import threading
import paramiko
import socket
import os
import logging
import sys
from typing import Optional, Tuple

# --- Setup basic logging ---
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Private Helper Functions ---

def _transfer_data(src, dest, transport_active_flag):
    """Pumps data between two sockets."""
    while transport_active_flag.is_set():
        try:
            data = src.recv(1024)
            if not data:
                break
            dest.sendall(data)
        except Exception:
            break
    logger.debug("Data transfer thread finished.")

def _handle_client_connection(local_conn, addr, ssh_transport, remote_host, remote_port, transport_active_flag):
    """Handles one client connection from start to finish."""
    remote_conn = None
    try:
        logger.info(f"Client connected from {addr}. Opening tunnel to {remote_host}:{remote_port}")
        remote_conn = ssh_transport.open_channel(
            "direct-tcpip",
            dest_addr=(remote_host, remote_port),
            src_addr=addr
        )
        
        # Start bidirectional data transfer
        t1 = threading.Thread(
            target=_transfer_data, args=(local_conn, remote_conn, transport_active_flag), daemon=True
        )
        t2 = threading.Thread(
            target=_transfer_data, args=(remote_conn, local_conn, transport_active_flag), daemon=True
        )
        t1.start()
        t2.start()
        
        # Wait for the client or server to close the connection
        t1.join()
        t2.join()

    except paramiko.SSHException as e:
        logger.warning(f"SSH channel error for client {addr}: {e}")
    except Exception:
        # Silently handle common connection resets
        pass
    finally:
        if remote_conn:
            remote_conn.close()
        local_conn.close()
        logger.info(f"Closed connection from {addr}.")


def _start_listener_loop(local_socket, ssh_transport, remote_host, remote_port, transport_active_flag):
    """
    This is the main loop that runs in a dedicated thread.
    It accepts new client connections and spawns threads for them.
    """
    try:
        while transport_active_flag.is_set():
            logger.debug(f"Listener loop waiting on port {local_socket.getsockname()[1]}...")
            # 1. Block and wait for a new client (e.g., langgraph) to connect
            local_conn, addr = local_socket.accept()
            
            # 2. Check if we're supposed to be shutting down
            if not transport_active_flag.is_set():
                local_conn.close()
                break

            # 3. Start a *new* thread to handle this specific client
            # This allows the loop to immediately go back to waiting for another client
            forward_thread = threading.Thread(
                target=_handle_client_connection,
                args=(local_conn, addr, ssh_transport, remote_host, remote_port, transport_active_flag),
                daemon=True
            )
            forward_thread.start()

    except OSError:
        logger.info("Listener socket was closed; shutting down listener loop.")
    except Exception as e:
        logger.error(f"Error in listener loop: {e}")
    finally:
        logger.info("Listener loop has stopped.")


# --- Public Function ---

def start_tunnel(
    user_name: str,
    compute_node_host: str,
    local_port_on_login_node: int,
    remote_port_on_compute_node: int,
    remote_dest_host: str = "localhost",
    ssh_key_path: os.PathLike = os.path.expanduser("~/.ssh/id_rsa")
) -> Optional[Tuple[paramiko.SSHClient, threading.Thread, threading.Event]]:
    """
    Starts a persistent, multi-client SSH port forward.

    This function:
    1. Connects to the `compute_node_host` via SSH.
    2. Opens a socket on `local_port_on_login_node` on the login node.
    3. Starts a listener thread to accept *multiple* client connections.
    4. Forwards each connection to `remote_dest_host`:`remote_port_on_compute_node`
       (relative to the compute node).

    Returns a tuple of (ssh_client, listener_thread, shutdown_flag) on success, 
    or None on failure. You *must* keep a reference to these objects.
    """
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    local_socket = None
    
    try:
        logger.info(f"Attempting to connect to {compute_node_host} as {user_name}...")
        
        # This is a more robust way to load keys.
        # Paramiko's connect() handles all key types (RSA, Ed25519, etc.)
        ssh_client.connect(
            hostname=compute_node_host,
            username=user_name,
            key_filename=str(ssh_key_path), # Let paramiko handle key type
            timeout=10
        )
        logger.info(f"SSH connection to {compute_node_host} established.")

        ssh_transport = ssh_client.get_transport()
        ssh_transport.set_keepalive(30) # ESSENTIAL for HPC firewalls

        # This flag is used to signal the listener thread to shut down
        transport_active_flag = threading.Event()
        transport_active_flag.set()

        local_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        local_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        local_socket.bind(('', local_port_on_login_node))
        local_socket.listen(5) # Listen for up to 5 queued connections
        
        logger.info(f"Local listener started on port {local_port_on_login_node}")

        # Start the one-and-only listener thread
        listener_thread = threading.Thread(
            target=_start_listener_loop,
            args=(
                local_socket, 
                ssh_transport, 
                remote_dest_host, 
                remote_port_on_compute_node,
                transport_active_flag
            ),
            daemon=True
        )
        listener_thread.start()

        # Return the objects the main script needs to manage the tunnel
        return (ssh_client, listener_thread, transport_active_flag, local_socket)

    except Exception as e:
        logger.error(f"Failed to start persistent tunnel: {e}", exc_info=True)
        if ssh_client:
            ssh_client.close()
        if local_socket:
            local_socket.close()
        return None

def stop_tunnel(
    ssh_client: paramiko.SSHClient,
    listener_thread: threading.Thread,
    shutdown_flag: threading.Event,
    local_socket: socket.socket
):
    """Gracefully shuts down the tunnel."""
    logger.info("Shutting down persistent tunnel...")
    
    # 1. Signal all threads to stop
    shutdown_flag.clear()
    
    # 2. Close the local socket to interrupt the blocking `accept()` call
    try:
        local_socket.close()
    except Exception:
        pass # Ignore errors if already closed
    
    # 3. Close the main SSH connection
    ssh_client.close()
    
    # 4. Wait for the listener thread to exit
    listener_thread.join(timeout=2.0)
    logger.info("Tunnel shutdown complete.")
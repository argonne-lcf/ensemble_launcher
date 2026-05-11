import math
import os
import subprocess
import sys


def _scatter_fn(node_local_cache: str, model: str, chunk_size: int):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    my_rank = comm.rank

    model_cache = os.path.join(
        node_local_cache, "hub", f"models--{model.replace('/', '--')}"
    )

    if my_rank == 0:
        files = []
        for dirpath, _, filenames in os.walk(model_cache):
            for filename in filenames:
                files.append(os.path.join(dirpath, filename))
        if not files:
            print(
                f"WARNING: No files found in {model_cache}. Nothing to sync.",
                file=sys.stderr,
            )
    else:
        files = None

    files = comm.bcast(files, root=0)

    if not files:
        return

    recv_buffer = bytearray(chunk_size)

    for file in files:
        if my_rank > 0:
            os.makedirs(os.path.dirname(file), exist_ok=True)

        if my_rank == 0:
            file_size = os.path.getsize(file)
        else:
            file_size = None

        file_size = comm.bcast(file_size, root=0)

        if file_size == 0:
            if my_rank > 0:
                open(file, "wb").close()
            continue

        nchunks = math.ceil(file_size / chunk_size)

        if my_rank == 0:
            f = open(file, "rb")
        else:
            f = open(file, "wb")

        try:
            for cid in range(nchunks):
                current_chunk_size = min(chunk_size, file_size - cid * chunk_size)

                if my_rank == 0:
                    chunk = f.read(current_chunk_size)
                    recv_buffer[:current_chunk_size] = chunk

                comm.Bcast([recv_buffer, current_chunk_size, MPI.BYTE], root=0)

                if my_rank > 0:
                    f.write(recv_buffer[:current_chunk_size])
        finally:
            f.close()

    if my_rank == 0:
        print(
            f"Successfully scattered {len(files)} files to {comm.size - 1} worker nodes."
        )


def sync_to_root(
    model: str,
    cache_dir: str,
    node_local_cache: str = "/tmp/model_cache",
    np: int = 16,
):
    src = os.path.join(cache_dir, "hub", f"models--{model.replace('/', '--')}")
    dst = os.path.join(node_local_cache, "hub", f"models--{model.replace('/', '--')}")

    os.makedirs(dst, exist_ok=True)

    cmd = ["mpirun", "-np", str(np), "-ppn", str(np), "dsync", f"{src}/", f"{dst}/"]
    p = subprocess.run(cmd, capture_output=True, text=True)

    if p.returncode != 0:
        raise RuntimeError(
            f"dsync failed (rc={p.returncode})\nstdout: {p.stdout}\nstderr: {p.stderr}"
        )

    return p


def scatter_from_root(
    model: str,
    node_local_cache: str,
    nnodes: int,
    ppn: int = 1,
    chunk_size: int = 100 * 1024 * 1024,
):
    cmd = [
        "mpirun",
        "-np", str(nnodes * ppn),
        "-ppn", str(ppn),
        sys.executable, "-c",
        f"from ensemble_launcher.inference.copy_model import _scatter_fn; "
        f"_scatter_fn({node_local_cache!r}, {model!r}, {chunk_size!r})",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)

    if p.returncode != 0:
        raise RuntimeError(
            f"scatter failed (rc={p.returncode})\nstdout: {p.stdout}\nstderr: {p.stderr}"
        )

    return p

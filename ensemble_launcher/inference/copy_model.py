import math
import os
import subprocess
import sys
from logging import Logger
from typing import Optional


def _scatter_fn(
    node_local_cache: str,
    model: str,
    chunk_size: int,
    ppn: int = 1,
    cache_modelinfo: bool = False,
):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    my_rank = comm.rank

    my_color = my_rank % ppn
    sub_comm = comm.Split(color=my_color, key=my_rank // ppn)
    sub_rank = sub_comm.rank
    is_node_lead = my_color == 0

    if cache_modelinfo:
        model_cache = node_local_cache
    else:
        model_cache = os.path.join(
            node_local_cache, "hub", f"models--{model.replace('/', '--')}"
        )

    if my_rank == 0:
        regular_files = []
        symlinks = []
        for dirpath, _, filenames in os.walk(model_cache):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.islink(filepath):
                    symlinks.append((filepath, os.readlink(filepath)))
                else:
                    regular_files.append(filepath)
        if not regular_files and not symlinks:
            print(
                f"WARNING: No files found in {model_cache}. Nothing to sync.",
                file=sys.stderr,
            )
    else:
        regular_files = None
        symlinks = None

    regular_files = comm.bcast(regular_files, root=0)
    symlinks = comm.bcast(symlinks, root=0)

    if not regular_files and not symlinks:
        sub_comm.Free()
        return

    recv_buffer = bytearray(chunk_size)

    for file in regular_files:
        if is_node_lead and sub_rank > 0:
            os.makedirs(os.path.dirname(file), exist_ok=True)

        if sub_rank == 0:
            file_size = os.path.getsize(file)
        else:
            file_size = None
        file_size = sub_comm.bcast(file_size, root=0)

        if file_size == 0:
            if is_node_lead and sub_rank > 0:
                open(file, "wb").close()
            if ppn > 1:
                comm.Barrier()
            continue

        base_part_size = file_size // ppn
        remainder = file_size % ppn
        if my_color < remainder:
            my_part_size = base_part_size + 1
            part_start = my_color * (base_part_size + 1)
        else:
            my_part_size = base_part_size
            part_start = (
                remainder * (base_part_size + 1)
                + (my_color - remainder) * base_part_size
            )

        if ppn > 1 and my_part_size > 0:
            part_file = f"{file}.__part{my_color}__"
        else:
            part_file = file

        nchunks = math.ceil(my_part_size / chunk_size) if my_part_size > 0 else 0

        if my_part_size > 0:
            if sub_rank == 0:
                f_in = open(file, "rb")
                f_in.seek(part_start)
            else:
                f_in = None

            if sub_rank > 0:
                os.makedirs(os.path.dirname(part_file), exist_ok=True)
                f_out = open(part_file, "wb")
            else:
                f_out = None

            try:
                for cid in range(nchunks):
                    current_chunk_size = min(
                        chunk_size, my_part_size - cid * chunk_size
                    )
                    if sub_rank == 0:
                        chunk = f_in.read(current_chunk_size)
                        recv_buffer[:current_chunk_size] = chunk
                    sub_comm.Bcast([recv_buffer, current_chunk_size, MPI.BYTE], root=0)
                    if sub_rank > 0:
                        f_out.write(recv_buffer[:current_chunk_size])
            finally:
                if f_in is not None:
                    f_in.close()
                if f_out is not None:
                    f_out.close()

        if ppn > 1:
            comm.Barrier()
            if is_node_lead and sub_rank > 0:
                with open(file, "wb") as f_merged:
                    for p in range(ppn):
                        pf = f"{file}.__part{p}__"
                        if os.path.exists(pf):
                            with open(pf, "rb") as pf_in:
                                while True:
                                    data = pf_in.read(chunk_size)
                                    if not data:
                                        break
                                    f_merged.write(data)
                            os.remove(pf)

    if is_node_lead and sub_rank > 0:
        for link_path, target in symlinks:
            os.makedirs(os.path.dirname(link_path), exist_ok=True)
            if os.path.lexists(link_path):
                os.remove(link_path)
            os.symlink(target, link_path)

    sub_comm.Free()

    if my_rank == 0:
        print(
            f"Successfully scattered {len(regular_files)} files and "
            f"{len(symlinks)} symlinks to {comm.size // ppn - 1} worker nodes."
        )


def sync_to_root(
    model: str,
    cache_dir: str,
    node_local_cache: str = "/tmp/model_cache",
    np: int = 16,
    logger: Logger = None,
    cache_modelinfo: bool = False,
) -> list:

    processes = []

    def _dsync(src, dst):
        os.makedirs(dst, exist_ok=True)

        cmd = ["mpirun", "-np", str(np), "-ppn", str(np), "dsync", f"{src}/", f"{dst}/"]
        p = subprocess.run(cmd, capture_output=True, text=True)

        if p.returncode != 0:
            if logger:
                logger.warning(
                    f"dsync failed (rc={p.returncode})\nstdout: {p.stdout}\nstderr: {p.stderr}"
                )
            else:
                print(
                    f"dsync failed (rc={p.returncode})\nstdout: {p.stdout}\nstderr: {p.stderr}"
                )
        return p

    src = os.path.join(cache_dir, "hub", f"models--{model.replace('/', '--')}")
    dst = os.path.join(node_local_cache, "hub", f"models--{model.replace('/', '--')}")
    processes.append(_dsync(src, dst))

    if cache_modelinfo:
        if logger:
            logger.info("Trying to dsync model infos, torch aot compile")

        for dirname in ["modelinfos", "torch_aot_compile", "torch_compile_cache"]:
            if os.path.exists(os.path.join(cache_dir, dirname)):
                if logger:
                    logger.info(f"Dsync {dirname}")
                src = os.path.join(cache_dir, dirname)
                dst = os.path.join(node_local_cache, dirname)
                processes.append(_dsync(src, dst))

    return processes


def scatter_from_root(
    model: str,
    node_local_cache: str,
    nnodes: int,
    ppn: int = 1,
    chunk_size: int = 100 * 1024 * 1024,
    logger: Logger = None,
    cpu_binding: Optional[str] = None,
):
    cmd = ["mpirun", "-np", str(nnodes * ppn), "-ppn", str(ppn)]
    if cpu_binding:
        cmd += [f"{cpu_binding}"]

    cmd += [
        sys.executable,
        "-c",
        f"from ensemble_launcher.inference.copy_model import _scatter_fn; "
        f"_scatter_fn({node_local_cache!r}, {model!r}, {chunk_size!r}, {ppn!r})",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if logger:
        logger.info(f"Scatter_from_root: Executing {cmd}")

    if p.returncode != 0:
        raise RuntimeError(
            f"scatter failed (rc={p.returncode})\nstdout: {p.stdout}\nstderr: {p.stderr}"
        )
    else:
        logger.debug(p.stdout)

    return p

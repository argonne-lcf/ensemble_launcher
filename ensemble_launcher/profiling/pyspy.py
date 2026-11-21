"""Attach py-spy to running orchestrator nodes, found through the checkpoint directory.

py-spy samples a process's stacks from outside it, so profiling needs no code in the
orchestrator and costs it almost nothing. It must run on the same host as the process it
profiles, though, and a large run spreads its nodes over hundreds of hosts. Every node records
its host and pid in ``{node_id}_meta.json`` under the checkpoint directory; this module reads
those, picks nodes by glob, and starts ``py-spy record`` on each node's host -- directly when
the node is local, over ssh otherwise.

Each session writes ``{output_dir}/{node_id}_pyspy.json`` (Chrome trace format) when its node
exits or when it is stopped, plus a ``{node_id}_pyspy.meta.json`` sidecar whose start time
lets :mod:`~ensemble_launcher.profiling.merge_traces` line the nodes up. py-spy's own output
goes to ``{node_id}_pyspy.log``.
"""

import fnmatch
import json
import logging
import os
import shlex
import shutil
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

from ensemble_launcher.orchestrator.discovery import NodeProcess, read_node_processes

logger = logging.getLogger(__name__)


def select_nodes(nodes: Iterable[NodeProcess], globs: Sequence[str]) -> List[NodeProcess]:
    """Nodes whose id matches any of ``globs``, sorted by node id."""
    return sorted(
        (n for n in nodes if any(fnmatch.fnmatchcase(n.node_id, g) for g in globs)),
        key=lambda n: n.node_id,
    )


def _short_host(hostname: str) -> str:
    return hostname.split(".")[0]


def is_local(hostname: str) -> bool:
    return _short_host(hostname) == _short_host(socket.gethostname())


@dataclass
class PySpySession:
    """One py-spy process attached to one node."""

    node: NodeProcess
    trace_path: str
    log_path: str
    popen: subprocess.Popen
    local: bool
    start_us: Optional[int] = None
    remote_pid: Optional[int] = None
    error: Optional[str] = field(default=None)

    @property
    def running(self) -> bool:
        return self.popen.poll() is None


class PySpyProfiler:
    """Profile selected orchestrator nodes with py-spy.

    Example, in a driver script right after ``EnsembleLauncher.start()``::

        with PySpyProfiler(ckpt_dir, nodes=["main", "main.m2*"]) as profiler:
            ...  # submit work; on exit, sessions are stopped and traces flushed

    Args:
        checkpoint_dir: The run's ``LauncherConfig.checkpoint_dir``.
        nodes: Node-id globs to profile, e.g. ``["main", "main.m2*"]``.
        output_dir: Where traces go. Must be on a filesystem every host can write, since
            remote py-spy processes write there directly.
        rate: Samples per second.
        idle: Keep samples of threads that are idle (waiting on I/O or a lock). Needed to see
            when the event loop is idle versus busy.
        gil: Keep only samples of threads holding the GIL.
        native: Include C frames, which is what names a long C call holding the loop.
        nonblocking: Do not pause the target while sampling; lower overhead, less accurate.
        duration: Stop each session after this many seconds; default is until its node exits.
        pyspy: py-spy executable. Defaults to the one on this host's PATH, resolved to an
            absolute path so remote hosts run the same binary from the shared filesystem.
        wait_timeout: How long :meth:`start` waits for every glob to match a checkpointed node.
        ssh_opts: Extra ``ssh`` options for remote hosts.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        nodes: Sequence[str] = ("main",),
        output_dir: str = "profiles",
        rate: int = 100,
        idle: bool = True,
        gil: bool = False,
        native: bool = False,
        nonblocking: bool = False,
        duration: Optional[int] = None,
        pyspy: Optional[str] = None,
        wait_timeout: float = 120.0,
        ssh_opts: Sequence[str] = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=30"),
    ):
        self.checkpoint_dir = checkpoint_dir
        self.globs = list(nodes)
        self.output_dir = os.path.abspath(output_dir)
        self.rate = rate
        self.idle = idle
        self.gil = gil
        self.native = native
        self.nonblocking = nonblocking
        self.duration = duration
        self.wait_timeout = wait_timeout
        self.ssh_opts = list(ssh_opts)
        found = pyspy or shutil.which("py-spy")
        if found is None:
            raise FileNotFoundError(
                "py-spy not found; install it with `pip install py-spy` "
                "(or the `profiling` extra) or pass pyspy=<path>"
            )
        self.pyspy = os.path.abspath(found) if os.path.sep in found else found
        self.sessions: List[PySpySession] = []

    # ------------------------------------------------------------------ #
    # Command construction (pure, so it can be tested without processes)  #
    # ------------------------------------------------------------------ #

    def trace_path(self, node_id: str) -> str:
        return os.path.join(self.output_dir, f"{node_id}_pyspy.json")

    def log_path(self, node_id: str) -> str:
        return os.path.join(self.output_dir, f"{node_id}_pyspy.log")

    def sidecar_path(self, node_id: str) -> str:
        return os.path.join(self.output_dir, f"{node_id}_pyspy.meta.json")

    def pyspy_args(self, node: NodeProcess) -> List[str]:
        args = [
            self.pyspy,
            "record",
            "--pid", str(node.pid),
            "--format", "chrometrace",
            "--output", self.trace_path(node.node_id),
            "--rate", str(self.rate),
        ]  # fmt: skip
        if self.idle:
            args.append("--idle")
        if self.gil:
            args.append("--gil")
        if self.native:
            args.append("--native")
        if self.nonblocking:
            args.append("--nonblocking")
        if self.duration is not None:
            args += ["--duration", str(self.duration)]
        return args

    def remote_command(self, node: NodeProcess) -> List[str]:
        """ssh command that runs py-spy on the node's host.

        The remote shell first prints its pid and clock, then ``exec``s py-spy so py-spy keeps
        that pid: the pid is what :meth:`stop` signals, and the clock reading is the trace's
        start time on the node's own clock.
        """
        script = (
            "echo $$ $(date +%s%N); "
            f"exec {shlex.join(self.pyspy_args(node))} "
            f">{shlex.quote(self.log_path(node.node_id))} 2>&1"
        )
        return ["ssh", *self.ssh_opts, node.hostname, script]

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def discover(self) -> List[NodeProcess]:
        """Wait until every glob matches a checkpointed node, or until ``wait_timeout``."""
        deadline = time.monotonic() + self.wait_timeout
        while True:
            procs = read_node_processes(self.checkpoint_dir) if os.path.isdir(
                self.checkpoint_dir
            ) else {}
            unmatched = [
                g
                for g in self.globs
                if not any(fnmatch.fnmatchcase(n, g) for n in procs)
            ]
            if not unmatched:
                return select_nodes(procs.values(), self.globs)
            if time.monotonic() >= deadline:
                logger.warning(
                    f"No checkpointed node matches {unmatched} in {self.checkpoint_dir} "
                    f"after {self.wait_timeout:.0f}s; profiling the nodes that did match"
                )
                return select_nodes(procs.values(), self.globs)
            time.sleep(1.0)

    def start(self) -> List[PySpySession]:
        """Attach py-spy to every selected node. Returns the sessions that started."""
        os.makedirs(self.output_dir, exist_ok=True)
        for node in self.discover():
            self.sessions.append(self._launch(node))

        # ssh prints the remote pid and clock as its first line; read them only after every
        # session is launched, so the hosts start up in parallel rather than one by one.
        for session in self.sessions:
            if not session.local:
                fields = session.popen.stdout.readline().split()
                try:
                    session.remote_pid = int(fields[0])
                    session.start_us = int(fields[1]) // 1000
                except (IndexError, ValueError):
                    # ssh failed before the shell ran, or the host's `date` lacks %N
                    # (BSD). The poll below reports the former; for the latter, the
                    # local clock is close enough.
                    if session.start_us is None:
                        session.start_us = time.time_ns() // 1000
            self._write_sidecar(session)

        # A session that cannot attach (ssh refused, py-spy missing, ptrace denied) exits
        # within moments; report it, but leave the other sessions running.
        time.sleep(1.0)
        for session in self.sessions:
            code = session.popen.poll()
            if code not in (None, 0):
                session.error = self._read_log_tail(session)
                logger.error(
                    f"py-spy for {session.node.node_id} on {session.node.hostname} "
                    f"exited with {code}: {session.error}"
                )
        started = [s for s in self.sessions if s.error is None]
        logger.info(
            f"py-spy attached to {len(started)}/{len(self.sessions)} nodes, "
            f"writing to {self.output_dir}"
        )
        return started

    def _launch(self, node: NodeProcess) -> PySpySession:
        trace_path, log_path = self.trace_path(node.node_id), self.log_path(node.node_id)
        if is_local(node.hostname):
            with open(log_path, "w") as log:
                start_us = time.time_ns() // 1000
                popen = subprocess.Popen(
                    self.pyspy_args(node), stdout=log, stderr=subprocess.STDOUT
                )
            return PySpySession(node, trace_path, log_path, popen, True, start_us)
        popen = subprocess.Popen(
            self.remote_command(node),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return PySpySession(node, trace_path, log_path, popen, False)

    def _write_sidecar(self, session: PySpySession) -> None:
        with open(self.sidecar_path(session.node.node_id), "w") as f:
            json.dump(
                {
                    "node_id": session.node.node_id,
                    "hostname": session.node.hostname,
                    "pid": session.node.pid,
                    "start_us": session.start_us,
                },
                f,
            )

    def _read_log_tail(self, session: PySpySession, limit: int = 2000) -> str:
        text = ""
        if not session.local and session.popen.stderr is not None:
            text = session.popen.stderr.read() or ""
        try:
            with open(session.log_path, "r") as f:
                text += f.read()
        except OSError:
            pass
        return text.strip()[-limit:]

    def wait(self, timeout: Optional[float] = None) -> None:
        """Block until every session has ended, i.e. until the profiled nodes exit."""
        deadline = None if timeout is None else time.monotonic() + timeout
        for session in self.sessions:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            session.popen.wait(timeout=remaining)

    def stop(self, timeout: float = 30.0) -> None:
        """Ask every running session to write its trace and exit.

        py-spy writes its trace on SIGINT. Remote sessions are signalled over ssh using the pid
        their shell reported; killing the local ssh client instead would hang up py-spy
        before it writes anything.
        """
        for session in self.sessions:
            if not session.running:
                continue
            if session.local:
                session.popen.send_signal(signal.SIGINT)
            elif session.remote_pid is not None:
                subprocess.run(
                    ["ssh", *self.ssh_opts, session.node.hostname,
                     f"kill -INT {session.remote_pid}"],
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    timeout=timeout,
                )  # fmt: skip
        deadline = time.monotonic() + timeout
        for session in self.sessions:
            try:
                session.popen.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                logger.warning(f"py-spy for {session.node.node_id} did not exit; killing it")
                session.popen.kill()

    def __enter__(self) -> "PySpyProfiler":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

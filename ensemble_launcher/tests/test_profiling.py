import json
import os
import shutil
import socket
import subprocess
import sys
import textwrap
import time

import pytest

from ensemble_launcher.checkpointing import CheckpointData
from ensemble_launcher.orchestrator.discovery import NodeProcess, read_node_processes
from ensemble_launcher.profiling import PySpyProfiler, merge_traces
from ensemble_launcher.profiling.merge_traces import node_events, node_pid, read_trace, select
from ensemble_launcher.profiling.pyspy import select_nodes

pytestmark = pytest.mark.core


def _write_meta(ckpt_dir, node_id, hostname, pid):
    sub = os.path.join(ckpt_dir, *node_id.split("."))
    os.makedirs(sub, exist_ok=True)
    meta = CheckpointData(timestamp=time.time(), node_id=node_id, hostname=hostname, pid=pid)
    with open(os.path.join(sub, f"{node_id}_meta.json"), "w") as f:
        f.write(meta.model_dump_json())


# ---------------------------------------------------------------------------
# Checkpoint metadata and discovery
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_checkpoint_meta_records_writer_pid():
    """The pid is taken when the metadata is built, not when the module was imported."""
    read_end, write_end = os.pipe()
    child = os.fork()
    if child == 0:
        os.write(write_end, str(CheckpointData(timestamp=0.0, node_id="x").pid).encode())
        os._exit(0)
    os.close(write_end)
    recorded = int(os.read(read_end, 64))
    os.waitpid(child, 0)
    assert recorded == child


def test_read_node_processes(tmp_path):
    ckpt = str(tmp_path)
    _write_meta(ckpt, "main", "host-a", 100)
    _write_meta(ckpt, "main.m0", "host-b", 200)
    _write_meta(ckpt, "main.m0.w0", "host-c", 300)
    os.makedirs(tmp_path / "main" / "m1")
    (tmp_path / "main" / "m1" / "main.m1_meta.json").write_text("{not json")  # mid-write

    procs = read_node_processes(ckpt)

    assert {n: (p.hostname, p.pid) for n, p in procs.items()} == {
        "main": ("host-a", 100),
        "main.m0": ("host-b", 200),
        "main.m0.w0": ("host-c", 300),
    }


def test_select_nodes():
    nodes = [NodeProcess(n, "h", 1, 0.0) for n in ["main", "main.m0", "main.m1", "main.m0.w0"]]
    assert [n.node_id for n in select_nodes(nodes, ["main"])] == ["main"]
    assert [n.node_id for n in select_nodes(nodes, ["main.m?"])] == ["main.m0", "main.m1"]
    assert [n.node_id for n in select_nodes(nodes, ["main", "*.w*"])] == ["main", "main.m0.w0"]


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------


def test_pyspy_command(tmp_path):
    profiler = PySpyProfiler(
        str(tmp_path), output_dir=str(tmp_path / "out"), pyspy="/opt/bin/py-spy",
        rate=50, native=True, gil=True, duration=30,
    )  # fmt: skip
    node = NodeProcess("main.m0", "x1000c0s0b0n0", 4242, 0.0)

    args = profiler.pyspy_args(node)
    assert args[:2] == ["/opt/bin/py-spy", "record"]
    assert args[args.index("--pid") + 1] == "4242"
    assert args[args.index("--format") + 1] == "chrometrace"
    assert args[args.index("--output") + 1] == str(tmp_path / "out" / "main.m0_pyspy.json")
    assert args[args.index("--rate") + 1] == "50"
    assert args[args.index("--duration") + 1] == "30"
    assert {"--idle", "--gil", "--native"} <= set(args)
    assert "--nonblocking" not in args

    remote = profiler.remote_command(node)
    assert remote[0] == "ssh" and remote[-2] == "x1000c0s0b0n0"
    assert "BatchMode=yes" in remote
    script = remote[-1]
    assert script.startswith("echo $$ $(date +%s%N); exec /opt/bin/py-spy record --pid 4242")
    assert script.endswith(f">{tmp_path / 'out' / 'main.m0_pyspy.log'} 2>&1")


def test_missing_pyspy_is_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="py-spy not found"):
        PySpyProfiler(str(tmp_path))


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


def _write_pyspy_output(out_dir, node_id, host, start_us, truncate=False):
    events = [
        {"name": "<module>", "cat": "py-spy", "ph": "B", "pid": 1234, "tid": 7, "ts": 0},
        {"name": "<module>", "cat": "py-spy", "ph": "E", "pid": 1234, "tid": 7, "ts": 500},
    ]
    text = json.dumps(events)
    if truncate:
        text = text[: text.rindex("},") + 2] + '{"name": "<mod'  # cut mid-event, no "]"
    (out_dir / f"{node_id}_pyspy.json").write_text(text)
    (out_dir / f"{node_id}_pyspy.meta.json").write_text(
        json.dumps({"node_id": node_id, "hostname": host, "pid": 1234, "start_us": start_us})
    )


def test_read_truncated_trace(tmp_path):
    _write_pyspy_output(tmp_path, "main", "h", 0, truncate=True)
    assert [e["ph"] for e in read_trace(str(tmp_path / "main_pyspy.json"))] == ["B"]


def test_merge_aligns_and_separates_nodes(tmp_path):
    _write_pyspy_output(tmp_path, "main", "host-a", 1_000_000)
    _write_pyspy_output(tmp_path, "main.m0", "host-b", 2_000_000)
    _write_pyspy_output(tmp_path, "main.m1", "host-c", 3_000_000)
    paths = [str(p) for p in tmp_path.glob("*_pyspy.json")]

    # Both nodes reported OS pid 1234; each gets its own stable pid and a labelled track.
    main = node_events(str(tmp_path / "main_pyspy.json"))
    assert main[0] == {
        "name": "process_name", "ph": "M", "pid": node_pid("main"),
        "args": {"name": "main (host-a)"},
    }  # fmt: skip
    assert {e["pid"] for e in main} == {node_pid("main")}
    # py-spy's relative timestamps land on the shared wall-clock timeline.
    assert [e["ts"] for e in main[1:]] == [1_000_000, 1_000_500]

    assert select(paths, ["main", "main.m1"]) == sorted(
        str(tmp_path / f"{n}_pyspy.json") for n in ["main", "main.m1"]
    )

    out = tmp_path / "merged.json"
    count = merge_traces(select(paths, None), str(out))
    merged = json.loads(out.read_text())
    assert count == len(merged) == 9
    assert {e["pid"] for e in merged} == {node_pid(n) for n in ["main", "main.m0", "main.m1"]}


# ---------------------------------------------------------------------------
# Sessions, against a fake py-spy (real py-spy needs root on macOS)
# ---------------------------------------------------------------------------

FAKE_PYSPY = """\
#!{python}
# Mimics `py-spy record`: runs until the target exits or SIGINT, then writes its trace.
import argparse, json, os, signal, time
parser = argparse.ArgumentParser()
parser.add_argument("cmd")
parser.add_argument("--pid", type=int)
parser.add_argument("--output")
args, _ = parser.parse_known_args()
stopped = []
signal.signal(signal.SIGINT, lambda *a: stopped.append(True))

def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False

start = time.monotonic()
while not stopped and alive(args.pid):
    time.sleep(0.05)
end = int((time.monotonic() - start) * 1e6)
with open(args.output, "w") as f:
    json.dump([
        {{"name": "<module>", "cat": "py-spy", "ph": "B", "pid": args.pid, "tid": 1, "ts": 0}},
        {{"name": "<module>", "cat": "py-spy", "ph": "E", "pid": args.pid, "tid": 1, "ts": end}},
    ], f)
"""

# Runs the "remote" command locally: drops the -o options and the host.
FAKE_SSH = """\
#!/bin/sh
while [ "$1" = "-o" ]; do shift 2; done
shift
exec sh -c "$*"
"""


@pytest.fixture
def fake_tools(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pyspy = bin_dir / "py-spy"
    pyspy.write_text(FAKE_PYSPY.format(python=sys.executable))
    ssh = bin_dir / "ssh"
    ssh.write_text(FAKE_SSH)
    for tool in (pyspy, ssh):
        tool.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return str(pyspy)


def _trace_ok(out_dir, node_id):
    events = json.loads((out_dir / f"{node_id}_pyspy.json").read_text())
    return [e["ph"] for e in events] == ["B", "E"]


@pytest.mark.parametrize("host", ["local", "remote"])
def test_session_ends_with_its_node(tmp_path, fake_tools, host):
    ckpt, out = tmp_path / "ckpt", tmp_path / "out"
    target = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        hostname = socket.gethostname() if host == "local" else "remote-host"
        _write_meta(str(ckpt), "main", hostname, target.pid)

        profiler = PySpyProfiler(str(ckpt), nodes=["main"], output_dir=str(out), wait_timeout=10)
        started = profiler.start()
        assert len(started) == 1 and started[0].local == (host == "local")
        if host == "remote":
            assert started[0].remote_pid is not None
        sidecar = json.loads((out / "main_pyspy.meta.json").read_text())
        assert sidecar["pid"] == target.pid and sidecar["start_us"] > 0
    finally:
        target.terminate()
        target.wait()  # reap it, so the profiler sees the node gone

    profiler.wait(timeout=10)
    assert _trace_ok(out, "main")


@pytest.mark.parametrize("host", ["local", "remote"])
def test_stop_flushes_running_session(tmp_path, fake_tools, host):
    ckpt, out = tmp_path / "ckpt", tmp_path / "out"
    target = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        hostname = socket.gethostname() if host == "local" else "remote-host"
        _write_meta(str(ckpt), "main", hostname, target.pid)
        with PySpyProfiler(str(ckpt), nodes=["main"], output_dir=str(out), wait_timeout=10) as p:
            assert all(s.running for s in p.sessions)
        assert not any(s.running for s in p.sessions)
        assert _trace_ok(out, "main")
    finally:
        target.kill()
        target.wait()


def test_failed_attach_does_not_stop_others(tmp_path, fake_tools):
    ckpt, out = tmp_path / "ckpt", tmp_path / "out"
    target = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        _write_meta(str(ckpt), "main.m0", socket.gethostname(), target.pid)
        _write_meta(str(ckpt), "main.m1", "remote-host", target.pid)
        profiler = PySpyProfiler(
            str(ckpt), nodes=["main.m*"], output_dir=str(out), wait_timeout=10,
            ssh_opts=("-o", "BatchMode=yes"),
        )  # fmt: skip
        # Make only the "remote" one fail, as an unreachable host would.
        profiler.remote_command = lambda node: ["sh", "-c", "echo boom >&2; exit 255"]
        started = profiler.start()
        assert [s.node.node_id for s in started] == ["main.m0"]
        failed = [s for s in profiler.sessions if s.error is not None]
        assert [s.node.node_id for s in failed] == ["main.m1"]
        assert "boom" in failed[0].error
        profiler.stop()
        assert _trace_ok(out, "main.m0")
    finally:
        target.kill()
        target.wait()


def test_unmatched_glob_times_out(tmp_path, fake_tools):
    ckpt = tmp_path / "ckpt"
    _write_meta(str(ckpt), "main", socket.gethostname(), os.getpid())
    profiler = PySpyProfiler(str(ckpt), nodes=["main", "main.m*"], wait_timeout=1.5)
    start = time.monotonic()
    assert [n.node_id for n in profiler.discover()] == ["main"]
    assert time.monotonic() - start >= 1.5


# ---------------------------------------------------------------------------
# Real py-spy, where it can attach (Linux; macOS needs root)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    shutil.which("py-spy") is None or sys.platform == "darwin",
    reason="needs py-spy and ptrace permission",
)
def test_real_pyspy(tmp_path):
    ckpt, out = tmp_path / "ckpt", tmp_path / "out"
    code = textwrap.dedent(
        """
        import time
        end = time.time() + 3
        while time.time() < end:
            sum(range(10000))
        """
    )
    target = subprocess.Popen([sys.executable, "-c", code])
    try:
        _write_meta(str(ckpt), "main", socket.gethostname(), target.pid)
        profiler = PySpyProfiler(str(ckpt), nodes=["main"], output_dir=str(out), wait_timeout=10)
        assert len(profiler.start()) == 1
        target.wait()
        profiler.wait(timeout=30)
    finally:
        target.kill()
    events = read_trace(str(out / "main_pyspy.json"))
    assert any(e.get("ph") == "B" for e in events)

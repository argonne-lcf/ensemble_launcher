"""Merge per-node py-spy traces into one Perfetto file.

:class:`~ensemble_launcher.profiling.pyspy.PySpyProfiler` leaves, per profiled node, a
``{node_id}_pyspy.json`` (Chrome trace format) and a ``{node_id}_pyspy.meta.json`` sidecar.
py-spy's raw output cannot simply be concatenated:

- its timestamps count from that py-spy's own start, so each trace is shifted by the start
  time recorded in its sidecar to put all nodes on one wall-clock timeline;
- its ``pid`` is the OS pid, which collides across hosts, so each node gets a stable pid
  derived from its node id, labelled with the node id and host;
- its ``tid`` is a raw ``pthread_t``, too large for Perfetto's JSON importer, which then
  folds all threads onto one track. Threads get small tids and role names instead
  (``event-loop``, ``thread-pool N``, ...), with the event loop listed first.

Merge even a single node's trace before opening it, or its threads will appear nested.

Thousands of nodes are far more than ui.perfetto.dev can load at once, so ``--nodes`` picks
the ones worth looking at::

    python -m ensemble_launcher.profiling.merge_traces --input-dir profiles \\
        --nodes main 'main.m2*' --output profiles/merged_perfetto.json
"""

import argparse
import fnmatch
import itertools
import json
import zlib
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional

TRACE_SUFFIX = "_pyspy.json"
SIDECAR_SUFFIX = "_pyspy.meta.json"


def node_pid(node_id: str) -> int:
    """Stable per-node pid, so merged traces never collide on OS pids from different hosts."""
    return zlib.crc32(node_id.encode()) & 0x7FFFFFFF


def node_id_of(trace_path: str) -> str:
    name = Path(trace_path).name
    return name[: -len(TRACE_SUFFIX)] if name.endswith(TRACE_SUFFIX) else Path(trace_path).stem


def read_trace(path: str) -> List[dict]:
    """Events of one trace file, tolerating a missing closing bracket or a cut-off last event."""
    text = Path(path).read_text().strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    body = text.rstrip("]").rstrip().rstrip(",")
    while body:
        try:
            return json.loads(body + "]")
        except json.JSONDecodeError:
            cut = body.rfind("},")
            if cut == -1:
                return []
            body = body[: cut + 1]
    return []


def read_sidecar(trace_path: str) -> dict:
    sidecar = Path(trace_path[: -len(TRACE_SUFFIX)] + SIDECAR_SUFFIX)
    try:
        return json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def node_events(trace_path: str, tids=None) -> List[dict]:
    """One node's events, on the shared timeline and under the node's own process track."""
    node_id = node_id_of(trace_path)
    meta = read_sidecar(trace_path)
    pid = node_pid(node_id)
    shift = meta.get("start_us") or 0
    host = meta.get("hostname")
    events = [
        {
            "name": "process_name",
            "ph": "M",
            "pid": pid,
            "args": {"name": f"{node_id} ({host})" if host else node_id},
        }
    ]
    raw = [e for e in read_trace(trace_path) if e]
    tid_map = _thread_tracks(raw, pid, tids if tids is not None else itertools.count(1), events)
    for event in raw:
        event = dict(event)
        event["pid"] = pid
        if "tid" in event:
            event["tid"] = tid_map[event["tid"]]
        if "ts" in event:
            event["ts"] += shift
        events.append(event)
    return events


# Thread roles, recognised from frames anywhere in a thread's samples. Checked in order.
_THREAD_ROLES = (
    ("event-loop", ("uv_run", "Runner.run", "run_forever", "<module>")),
    ("thread-pool", ("_WorkItem.run", "_worker")),
    ("mp-queue-feeder", ("Queue._feed",)),
)


def _thread_tracks(raw: List[dict], pid: int, tids, events: List[dict]) -> dict:
    """Map py-spy's thread ids to small unique tids and name each thread by its role.

    py-spy records raw ``pthread_t`` values, which exceed 32 bits on Linux. Perfetto's JSON
    importer cannot keep such ids apart and folds every thread onto the process's main
    track, so their stacks look nested inside one another.
    """
    frames: dict = {}
    for e in raw:
        if "tid" in e:
            frames.setdefault(e["tid"], set()).add(e.get("name"))
    labelled = []
    for orig, names in frames.items():
        rank, role = next(
            ((i, r) for i, (r, marks) in enumerate(_THREAD_ROLES) if names & set(marks)),
            (len(_THREAD_ROLES), "thread"),
        )
        labelled.append((rank, orig, role))
    labelled.sort()

    tid_map = {}
    seen = Counter()
    for rank, orig, role in labelled:
        tid_map[orig] = next(tids)
        seen[role] += 1
        name = role if role == "event-loop" else f"{role} {seen[role]}"
        events.append(
            {"name": "thread_name", "ph": "M", "pid": pid, "tid": tid_map[orig],
             "args": {"name": f"{name} (pthread {orig})"}}
        )  # fmt: skip
        events.append(
            {"name": "thread_sort_index", "ph": "M", "pid": pid, "tid": tid_map[orig],
             "args": {"sort_index": rank}}
        )  # fmt: skip
    return tid_map


def select(paths: Iterable[str], nodes: Optional[List[str]]) -> List[str]:
    """Keep the traces whose node id matches any of the ``nodes`` globs (all if none)."""
    paths = sorted(paths)
    if not nodes:
        return paths
    return [p for p in paths if any(fnmatch.fnmatchcase(node_id_of(p), g) for g in nodes)]


def merge_traces(paths: List[str], output: str) -> int:
    """Write the events of ``paths`` to ``output``; returns the number of events written."""
    count = 0
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as out:
        out.write("[\n")
        tids = itertools.count(1)  # unique across nodes, not just within one
        for path in paths:
            for event in node_events(path, tids):
                out.write((",\n" if count else "") + json.dumps(event, separators=(",", ":")))
                count += 1
        out.write("\n]\n")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge per-node py-spy traces")
    parser.add_argument("--input-dir", default="profiles")
    parser.add_argument("--output", default="profiles/merged_perfetto.json")
    parser.add_argument(
        "--nodes",
        nargs="*",
        help="node-id globs to include, e.g. main 'main.m2*' (default: all)",
    )
    args = parser.parse_args()

    paths = select((str(p) for p in Path(args.input_dir).glob(f"*{TRACE_SUFFIX}")), args.nodes)
    if not paths:
        print(f"No traces matching {args.nodes or '*'} in {args.input_dir}")
        return
    count = merge_traces(paths, args.output)
    print(f"Merged {len(paths)} traces ({count} events) into {args.output}")


if __name__ == "__main__":
    main()

"""Merge per-node py-spy traces into one Perfetto file.

:class:`~ensemble_launcher.profiling.pyspy.PySpyProfiler` leaves, per profiled node, a
``{node_id}_pyspy.json`` (Chrome trace format) and a ``{node_id}_pyspy.meta.json`` sidecar.
py-spy's raw output cannot simply be concatenated:

- its timestamps count from that py-spy's own start, so each trace is shifted by the start
  time recorded in its sidecar to put all nodes on one wall-clock timeline;
- its ``pid`` is the OS pid, which collides across hosts, so each node gets a stable pid
  derived from its node id, labelled with the node id and host.

Thousands of nodes are far more than ui.perfetto.dev can load at once, so ``--nodes`` picks
the ones worth looking at::

    python -m ensemble_launcher.profiling.merge_traces --input-dir profiles \\
        --nodes main 'main.m2*' --output profiles/merged_perfetto.json
"""

import argparse
import fnmatch
import json
import zlib
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


def node_events(trace_path: str) -> List[dict]:
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
    for event in read_trace(trace_path):
        if not event:
            continue
        event = dict(event)
        event["pid"] = pid
        if "ts" in event:
            event["ts"] += shift
        events.append(event)
    return events


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
        for path in paths:
            for event in node_events(path):
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

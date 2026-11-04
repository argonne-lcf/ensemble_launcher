# CLI Reference

Ensemble Launcher provides the `el` command-line interface for launching ensembles without writing Python scripts.

## Commands

```bash
el --help
el start --help
el stop --help
el profile --help
```

## `el start`

Launch an ensemble from a JSON configuration file.

```bash
el start ENSEMBLE_FILE [OPTIONS]
```

In normal mode, it blocks until all tasks finish and writes `results.json`. In cluster mode (when the launcher config has `"cluster": true`), it starts the orchestrator in the background and returns immediately.

### Options

| Option | Description |
|---|---|
| `--system-config-file` | Path to system configuration JSON |
| `--launcher-config-file` | Path to launcher configuration JSON |
| `--nodes-str` | Comma-separated compute nodes, e.g. `"node-001,node-002"` |

### Examples

**Basic execution:**
```bash
el start my_ensemble.json
```

**With custom configurations:**
```bash
el start my_ensemble.json \
    --system-config-file system.json \
    --launcher-config-file launcher.json
```

**Specify compute nodes:**
```bash
el start my_ensemble.json \
    --nodes-str "node-001,node-002,node-003,node-004"
```

**Cluster mode:**
```bash
el start my_ensemble.json --launcher-config-file cluster_launcher.json
# Returns immediately; orchestrator runs in background

# ... submit tasks from Python ...

el stop
```

## `el stop`

Send `SIGTERM` to a cluster-mode orchestrator started with `el start`, triggering graceful shutdown.

```bash
el stop
```

The PID of the background process is stored in `.el_launcher.pid` in the working directory.

## `el profile`

Attach [py-spy](https://github.com/benfred/py-spy) to running orchestrator nodes and record
what each one is doing over time, as a Perfetto-viewable timeline. Install it with
`pip install "ensemble_launcher[profiling]"`.

```bash
el profile --checkpoint-dir CKPT_DIR [--node GLOB ...] [OPTIONS]
```

Every node records its host and pid in the checkpoint directory, so the run needs a
`checkpoint_dir` (always the case in cluster mode). `el profile` waits for the matching nodes
to appear there, then starts `py-spy record` on each node's host: directly for nodes on the
local host, over `ssh` for the rest. Run it from inside the job, so the compute nodes are
reachable.

Each session writes its trace when its node exits, or when `el profile` is interrupted
(Ctrl-C or `SIGTERM`):

| File | Contents |
|---|---|
| `profiles/<node>_pyspy.json` | py-spy samples, Chrome trace format |
| `profiles/<node>_pyspy.meta.json` | Host, pid and start time, used to merge |
| `profiles/<node>_pyspy.log` | py-spy's own output, for diagnosing a failed attach |

### Options

| Option | Default | Description |
|---|---|---|
| `--checkpoint-dir` | required | The run's `LauncherConfig.checkpoint_dir` |
| `--node` | `main` | Node-id glob; repeat for several, e.g. `--node main --node 'main.m2*'` |
| `--output-dir` | `profiles` | Where traces go; must be on a filesystem every host can write |
| `--rate` | `100` | Samples per second |
| `--native` | off | Include C frames: names a long C call (e.g. unpickling) holding the event loop |
| `--gil` | off | Keep only samples of threads holding the GIL |
| `--idle / --no-idle` | on | Keep samples of idle threads, which shows when the event loop is idle |
| `--duration` | until nodes exit | Seconds to record |
| `--wait-timeout` | `120` | Seconds to wait for the nodes to checkpoint |
| `--pyspy` | from `PATH` | py-spy executable; use one on a shared filesystem so every host can run it |

### Example

In a job script, after starting a cluster:

```bash
el start ensemble.json --launcher-config-file cluster_launcher.json
el profile --checkpoint-dir "$CKPT_DIR" --node main --node 'main.m*' --native &
# ... submit work ...
el stop
wait   # py-spy sessions finish as their nodes exit

python -m ensemble_launcher.profiling.merge_traces --input-dir profiles \
    --nodes main 'main.m1*' --output profiles/merged_perfetto.json
```

Open `merged_perfetto.json` at [ui.perfetto.dev](https://ui.perfetto.dev). Each node is one
process track labelled with its node id and host, on a shared wall-clock timeline. Merge a
subset with `--nodes`: thousands of nodes are more than the UI can load at once.

The same is available from Python:

```python
from ensemble_launcher.profiling import PySpyProfiler

with PySpyProfiler(ckpt_dir, nodes=["main", "main.m2*"], native=True):
    ...  # traces are flushed when the block exits
```

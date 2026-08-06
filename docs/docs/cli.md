# CLI Reference

Ensemble Launcher provides the `el` command-line interface for launching ensembles without writing Python scripts.

## Commands

```bash
el --help
el start --help
el stop --help
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

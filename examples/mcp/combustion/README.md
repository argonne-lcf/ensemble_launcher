# Combustion MCP Example

Runs `compute_flame_speed` (Cantera 1D freely-propagating flame) as an ensemble MCP tool served over HTTP, so it can be used directly from Claude Code or any other MCP client.

## Requirements

```bash
pip install cantera ensemble_launcher
```

## Run the MCP server locally

```bash
cd examples/mcp/combustion
python3 start_mcp_http.py
```

The server starts on `http://localhost:8295/mcp` and exposes one tool:

| Tool | Description |
|---|---|
| `ensemble_compute_flame_speed` | Compute flame speed for a batch of (P, T, phi) inputs |

## Add to Claude Code

```bash
claude mcp add combustion --transport http http://localhost:8295/mcp
```

Verify it is registered:

```bash
claude mcp list
```

Claude can now call `ensemble_compute_flame_speed` directly. Example prompt:

> Compute the methane/air flame speed at P=1 atm, T=300 K for equivalence ratios 0.7, 0.9, 1.0, 1.1, 1.3.

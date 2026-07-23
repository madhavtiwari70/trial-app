# Divi Demo Console

A Streamlit app for card #208 ("Restructure Divi-Demos to become a demo framework"):
pick a demo, edit its data file, click Run, get real results back from a real
Divi execution. No notebook, no code changes.

Built on top of the [Streamlit blank-app template](https://github.com/streamlit/blank-app-template).

## Status

| Demo | Category | Status |
|---|---|---|
| Spin Dynamics (TFIM) | Time Evolution | ✅ ready — wired + tested |
| Cluster MaxCut | QAOA | ✅ ready — wired + tested |
| Molecular Ground State | Custom VQE | ✅ ready — wired + tested |
| Travelling Salesman | QAOA · QUBO | 🚧 not refactored yet |
| Portfolio Optimization | QAOA · PCE | 🚧 not refactored yet |
| Economic Load Dispatch | Optimization · PCE-VQE | 🚧 not refactored yet |
| Minimum Birkhoff Decomposition | Optimization | 🚧 not refactored yet |
| Quantum-Guided Cluster | QAOA | 🚧 not refactored yet |
| Qiskit/PennyLane → Divi | — | 🚧 doesn't exist yet in divi-demos at all |

All three "ready" demos were run end-to-end against the real `qoro-divi`
package (local `QiskitSimulator` backend) during development — not mock data.

### How to run it on your own machine

Prerequisite: install `uv` if you don't already have it.

```
$ curl -LsSf https://astral.sh/uv/install.sh | sh
```

1. Sync the dependencies

   ```
   $ uv sync
   ```

2. Run the app

   ```
   $ uv run streamlit run streamlit_app.py
   ```

If a demo's data file sets `backend.use_cloud: true`, set `QORO_API_KEY` in
your environment first (get one at https://dash.qoroquantum.net).

## How it's structured

```
streamlit_app.py                           <- Streamlit UI + demo registry
data/
  spin_dynamics.yaml
  cluster_maxcut.yaml
  molecular_ground_state.yaml               <- edit these to change each demo
demos/
  spin_dynamics/spin_dynamics_lib.py        <- run_from_config(cfg)
  cluster_maxcut/cluster_maxcut_lib.py      <- run_from_config(cfg)
  cluster_maxcut/utils.py                   <- unchanged graph-gen helper
  molecular_ground_state/molecular_ground_state_lib.py  <- run_from_config(cfg)
```

## How to add the remaining demos

Each follows the same pattern:
1. Copy the demo's original script from `divi-demos/<name>/` into `demos/<name>/`.
2. Pull every hardcoded constant into `data/<name>.yaml`.
3. Wrap the logic in `run_from_config(cfg, progress_callback=None)` returning
   a small results dataclass (figures + key metrics).
4. Add one entry to `DEMOS` in `streamlit_app.py` with `"status": "ready"`,
   and one `elif selected_label == "..."` branch to call it and render output.
5. Smoke-test standalone (shrink problem size) before wiring into the UI.
6. Run `uv sync` again if new dependencies were added.

## Deploying for real customer-call use

Push to this GitHub repo, deploy on Streamlit Community Cloud (free tier)
pointed at `streamlit_app.py`, with `QORO_API_KEY` set as a secret. Gives a
single shareable link, no local setup for whoever's running the call.

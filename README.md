# Divi Molecular Ground State Demo

A Streamlit app wrapping the Molecular Ground State (VQE) demo from card #208:
edit the data file, click Run, get a real ground-state energy back from a
real Divi execution. No notebook, no code changes.

Built on top of the [Streamlit blank-app template](https://github.com/streamlit/blank-app-template).

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

If `data/molecular_ground_state.yaml` sets `backend.use_cloud: true`, set
`QORO_API_KEY` in your environment first (get one at https://dash.qoroquantum.net).

## How it's structured

```
streamlit_app.py                                       <- Streamlit UI
data/
  molecular_ground_state.yaml                           <- edit this to change the demo
demos/
  molecular_ground_state/molecular_ground_state_lib.py  <- run_from_config(cfg), the Divi logic
```

## What's in the data file

- `molecule.symbols` / `molecule.coordinates` — the atoms and geometry (e.g. swap H2 for a different molecule)
- `vqe.n_layers` — ansatz depth
- `vqe.optimizer_method` — COBYLA, Nelder-Mead, or L-BFGS-B
- `vqe.max_iterations`, `vqe.seed`
- `backend.use_cloud` — false runs locally, true runs on QoroService

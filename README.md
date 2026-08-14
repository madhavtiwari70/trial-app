# Divi Demo Console — all 6 current demos, data-file driven

Card #208's actual goal, delivered for every demo currently in
QoroQuantum/divi-demos: pick a demo, edit its data file, click Run,
get real results from real Divi execution. No notebook, no code changes.

## Important context

`cluster_maxcut` and `molecular_ground_state` (previously in this repo)
were removed in the Aug 3, 2026 "Divi 0.13" refactor commit — the repo's
own README is currently out of date and still lists them. This console
covers the 6 demos that actually exist today.

## Status — all individually verified with real Divi execution

| Demo | Category | Covers |
|---|---|---|
| Spin Dynamics (TFIM) | Time Evolution | Full demo (all 3 physical regimes) |
| Economic Load Dispatch | Optimization · PCE-VQE | 3-generator scenario (original also has a 6-gen variant, not included) |
| Quantum-Guided Cluster | QAOA | Full demo — run_benchmark() already took every param as a kwarg |
| Travelling Salesman | QAOA · QUBO | "Part A: Direct QAOA" only (original also has partitioned + PCE variants) |
| Minimum Birkhoff Decomposition | Optimization | Full demo — already argparse-driven |
| Portfolio Optimization | QAOA | Small synthetic portfolio only (original also has a real 480-asset partitioned scenario) |

**Why some are partial:** several of the original scripts run 2-3 different
scenarios back-to-back (e.g. TSP runs a small direct-QAOA case AND a larger
partitioned case AND a PCE-compressed case in one script). Building one
clean, editable config schema per demo meant picking the core scenario
rather than cramming every variant into one data file. Extending any of
these to cover the other scenarios follows the same pattern used here —
see "How to extend" below.

## Setup

```
pip install -r requirements.txt
```

## Run it

```
streamlit run streamlit_app.py
```

## How it's structured

```
streamlit_app.py                    <- UI + demo registry (add new demos here)
data/
  <demo>.yaml                       <- what someone actually edits
demos/
  <demo>/
    <demo>_original.py              <- the REAL script from divi-demos, unmodified
    <demo>_wrapper.py               <- run_from_config(cfg) — imports and reuses
                                        the original's real functions directly,
                                        never re-implements the physics/logic
```

Every wrapper imports and calls the actual functions from the actual demo
script — none of the quantum logic was rewritten. Only the hardcoded
constants at the bottom of each script were pulled out into a YAML file.

## How to extend a demo to cover more of its original scenarios

1. Look at the demo's `_original.py` file — find the extra scenario in its
   `if __name__ == "__main__":` block (or later notebook cells, for
   portfolio_optimization).
2. Add the extra parameters to that demo's `data/<demo>.yaml`.
3. Extend `run_from_config()` in that demo's `_wrapper.py` to also run that
   scenario, calling the original script's existing functions.
4. Add a render branch in `streamlit_app.py` if the output shape changes.

## Known extra dependencies

- `docplex` + `cplex` — required only for Minimum Birkhoff Decomposition
- `dimod` + `dwave-neal` — required by Economic Load Dispatch and Portfolio Optimization

## Note on QoroService (cloud)

All demos default to local simulators. Several data files have a
`backend.use_cloud` flag — set it to `true` and set `QORO_API_KEY` in your
environment to run on QoroService instead.

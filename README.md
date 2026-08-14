# Divi Spin Dynamics Demo

A Streamlit app wrapping the Spin Dynamics (TFIM) demo from divi-demos:
edit the data file, click Run, get real magnetization trajectories back
from a real Divi execution. No notebook, no code changes.

### How to run it on your own machine

```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run streamlit run streamlit_app.py
```

If `data/spin_dynamics.yaml` sets `backend.use_cloud: true`, set
`QORO_API_KEY` in your environment first.

## How it's structured

```
streamlit_app.py
data/
  spin_dynamics.yaml                    <- edit this to change the demo
demos/
  spin_dynamics/
    spin_dynamics_original.py           <- the real script from divi-demos, unmodified
    spin_dynamics_wrapper.py            <- run_from_config(cfg), reuses the original's real functions
    plotting.py                         <- the real plotting helpers, unmodified
```

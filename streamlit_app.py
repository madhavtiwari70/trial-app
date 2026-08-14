"""
Divi Demo Console — Spin Dynamics (TFIM)

Edit the data file below and click Run to execute a real Divi program.
No code changes needed to change the physical regimes or backend.

Run locally with:  uv run streamlit run streamlit_app.py
"""

import sys
import os
import yaml
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "demos", "spin_dynamics"))
from spin_dynamics_wrapper import run_from_config

st.set_page_config(page_title="Divi Demo Console — Spin Dynamics", layout="wide")

st.title("Divi Demo Console")
st.subheader("Spin Dynamics (TFIM)")
st.caption("Time Evolution · edit the data file, click Run, no code changes.")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "spin_dynamics.yaml")
with open(DATA_PATH) as f:
    raw_yaml = f.read()

col_config, col_results = st.columns([1, 1.6])

with col_config:
    st.markdown("**1. Configure** — edit the data file, no code involved.")
    edited_yaml = st.text_area("data/spin_dynamics.yaml", raw_yaml, height=420)
    run_clicked = st.button("Run demo", type="primary", use_container_width=True)

with col_results:
    st.markdown("**2. Output**")

    if not run_clicked:
        st.info("Edit the config and click **Run demo** to execute the real Divi program.")
    else:
        try:
            cfg = yaml.safe_load(edited_yaml)
        except yaml.YAMLError as e:
            st.error(f"Couldn't parse the data file: {e}")
            st.stop()

        status_box = st.empty()

        def on_progress(*args):
            if len(args) == 1:
                status_box.info(args[0])
            else:
                i, n, name = args
                status_box.info(f"Running: {name} ({i + 1}/{n})")

        with st.spinner("Executing on backend…"):
            results = run_from_config(cfg, progress_callback=on_progress)

        status_box.empty()

        backend_label = "QoroService (cloud)" if cfg["backend"]["use_cloud"] else "Local simulator"
        st.success(f"Ran {len(results)} experiment(s) on **{backend_label}**")

        for r in results:
            st.markdown(f"**{r['name']}**")
            st.pyplot(r["fig_dynamics"])
            st.pyplot(r["fig_error"])
            m1, m2 = st.columns(2)
            m1.metric("Exact Trotterization runtime", f"{r['runtime_exact_s']:.2f} s")
            m2.metric("QDrift runtime", f"{r['runtime_qdrift_s']:.2f} s")
            st.divider()
